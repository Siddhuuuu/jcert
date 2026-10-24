#!/usr/bin/env python3
"""JCert real-lens diagnostics: how identifiable and stable is the J-lens dictionary?

Lens vectors at layer l: u_t = J_l^T w_t, with w_t the unembedding row of token t.
In code that is the rows of  W_U[tokens] @ J_l  (orientation as in the nnsight reproduction).

Per layer this reports:
  * dictionary geometry: mutual coherence mu, Welch floor, certified k = ceil(.5(1+1/mu))-1,
    most confusable token pairs, sigma_min of random column subsets and of
    nearest-neighbour clusters (the realistic confusable sets)
  * spectrum of J_l - I: stable rank, r90 (rank holding 90% of the energy), relative tail
    energy beyond rank 16/64 (is J - I low-rank? this decides whether randomized SVD helps)
  * baselines: Gaussian null dictionary of the same shape, and the logit lens (J = I)

Usage:
  python jcert_real_lens.py --selftest                      # numpy only, synthetic data
  python jcert_real_lens.py --list-lenses                   # files on neuronpedia/jacobian-lens
  python jcert_real_lens.py --model gpt2 --hf-lens gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt
  python jcert_real_lens.py --model gpt2 --lens my_lens.pt  # local artifact (dict with key "J")
Real runs need torch, transformers, huggingface_hub (matplotlib optional).
Coherence on a vocabulary subset under-estimates full-vocabulary coherence (fewer pairs);
use --full to use every token (slow on big vocabularies).
"""
import argparse
import csv
import math
import os
import sys
import tempfile
import time

import numpy as np

KS = (2, 8, 25)  # certified-sparsity levels; subsets have 2k columns


# ------------------------------------------------------------------ core math (numpy only)
def unit_rows(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def welch(d, n):
    return math.sqrt((n - d) / (d * (n - 1))) if n > d else 0.0


def k_cert(mu):
    return max(int(math.ceil(0.5 * (1 + 1 / mu))) - 1, 0)


def coherence_rows(R, block=2048, top=5):
    """R: n x d with unit rows. Returns (mu, top pairs [(cos, i, j)])."""
    n = R.shape[0]
    cand = []
    cols = np.arange(n)[None, :]
    for s in range(0, n, block):
        G = np.abs(R[s:s + block] @ R.T)
        rows = np.arange(s, min(s + block, n))
        G[cols <= rows[:, None]] = 0.0  # keep i < j only
        flat = G.ravel()
        idx = np.argpartition(flat, -top)[-top:] if flat.size > top else np.arange(flat.size)
        cand += [(float(flat[t]), int(rows[t // n]), int(t % n)) for t in idx]
    cand.sort(reverse=True)
    return cand[0][0], cand[:top]


def smin_random(R, k, m, rng):
    s, n = 2 * k, R.shape[0]
    v = [np.linalg.svd(R[rng.choice(n, s, replace=False)], compute_uv=False)[-1] for _ in range(m)]
    return float(np.median(v)), float(np.percentile(v, 5))


def smin_cluster(R, k, seeds, rng):
    """sigma_min of a token plus its 2k-1 most-aligned neighbours (confusable sets)."""
    s, n = 2 * k, R.shape[0]
    sid = rng.choice(n, seeds, replace=False)
    C = np.abs(R[sid] @ R.T)
    C[np.arange(seeds), sid] = -1.0
    v = []
    for a in range(seeds):
        nb = np.argpartition(C[a], -(s - 1))[-(s - 1):]
        v.append(np.linalg.svd(R[np.concatenate([[sid[a]], nb])], compute_uv=False)[-1])
    return float(np.median(v)), float(np.percentile(v, 5))


def dict_stats(R, rng, m=100, seeds=60):
    n, d = R.shape
    mu, pairs = coherence_rows(R)
    out = {"mu": mu, "welch": welch(d, n), "k_cert": k_cert(mu), "pairs": pairs}
    for k in KS:
        if 2 * k <= min(d, n):
            out[f"rand_k{k}"], out[f"rand5_k{k}"] = smin_random(R, k, m, rng)
            out[f"clus_k{k}"], out[f"clus5_k{k}"] = smin_cluster(R, k, seeds, rng)
    return out


def spectrum_of(A):
    s = np.linalg.svd(A, compute_uv=False).astype(np.float64)
    e = s ** 2
    tot = e.sum()
    cum = np.cumsum(e) / tot
    d = A.shape[0]
    return {"dist_from_id": math.sqrt(tot / d), "stable_rank": float(tot / e[0]),
            "r90": int(np.searchsorted(cum, 0.9) + 1), "r90_frac": float((np.searchsorted(cum, 0.9) + 1) / d),
            "tail16": math.sqrt(max(1 - cum[min(15, d - 1)], 0.0)),
            "tail64": math.sqrt(max(1 - cum[min(63, d - 1)], 0.0))}


def analyze(J, W, n_tokens=8000, subset="random", seed=0, layers=None, log=print):
    rng = np.random.default_rng(seed)
    V, d = W.shape
    n = V if (n_tokens <= 0 or n_tokens >= V) else n_tokens
    ids = np.arange(n) if subset == "first" else np.sort(rng.choice(V, n, replace=False))
    Ws = np.ascontiguousarray(W[ids], dtype=np.float32)
    log(f"d_model={d}, vocab={V}, using n={n} tokens ({subset}), layers={sorted(J)}")
    base = {"gaussian null": dict_stats(unit_rows(rng.standard_normal((n, d)).astype(np.float32)), rng),
            "logit lens (J=I)": dict_stats(unit_rows(Ws), rng)}
    null_r90 = spectrum_of(rng.standard_normal((d, d)).astype(np.float32))["r90_frac"]
    rows = []
    for l in sorted(J):
        if layers and l not in layers:
            continue
        t0 = time.time()
        Jl = J[l].astype(np.float32)
        st = spectrum_of(Jl - np.eye(d, dtype=np.float32))
        st.update(dict_stats(unit_rows(Ws @ Jl), rng))
        st["layer"] = l
        rows.append(st)
        log(f"  layer {l:3d} done in {time.time() - t0:5.1f}s")
    return rows, base, null_r90, ids


def print_report(rows, base, null_r90, ids, tok=None):
    print("\nlayer | ||J-I||/sqrt(d) | r90/d  tail16 | mu     k_cert | clus s_min(k=8) | rand s_min(k=8)")
    for r in rows:
        print(f"{r['layer']:5d} | {r['dist_from_id']:15.3f} | {r['r90_frac']:5.2f}  {r['tail16']:6.3f} | "
              f"{r['mu']:.3f}  {r['k_cert']:6d} | {r.get('clus_k8', float('nan')):15.3f} | "
              f"{r.get('rand_k8', float('nan')):.3f}")
    for name, b in base.items():
        print(f"{name:>16}: mu={b['mu']:.3f} (Welch floor {b['welch']:.3f}), certified k<={b['k_cert']}, "
              f"clus s_min(k=8)={b.get('clus_k8', float('nan')):.3f}, rand s_min(k=8)={b.get('rand_k8', float('nan')):.3f}")
    print(f"Gaussian-matrix r90/d for reference: {null_r90:.2f} (structureless matrix; much smaller = low-rank)")
    worst = max(rows, key=lambda r: r["mu"])
    dec = (lambda i: repr(tok.decode([int(ids[i])]))) if tok is not None else (lambda i: str(int(ids[i])))
    print(f"most confusable pairs at layer {worst['layer']} (cosine, token a, token b):")
    for c, i, j in worst["pairs"]:
        print(f"   {c:.3f}  {dec(i)}  {dec(j)}")


def write_csv(rows, base, path):
    keys = [k for k in rows[0] if k != "pairs"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        for name, b in base.items():
            f.write(f"# baseline {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in b.items() if k != "pairs") + "\n")


def make_plot(rows, base, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plot")
        return
    L = [r["layer"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(L, [r["mu"] for r in rows], "o-", label="J-lens")
    ax[0].axhline(base["gaussian null"]["mu"], ls="--", c="gray", label="Gaussian null")
    ax[0].axhline(base["logit lens (J=I)"]["mu"], ls=":", c="k", label="logit lens")
    ax[0].set(title="mutual coherence", xlabel="layer")
    ax[0].legend()
    if "clus_k8" in rows[0]:
        ax[1].semilogy(L, [r["clus_k8"] for r in rows], "o-", label="neighbour clusters")
        ax[1].semilogy(L, [r["rand_k8"] for r in rows], "s-", label="random subsets")
    ax[1].set(title="median sigma_min of 16 columns", xlabel="layer")
    ax[1].legend()
    ax[2].plot(L, [r["r90_frac"] for r in rows], "o-", label="r90/d")
    ax[2].plot(L, [r["dist_from_id"] for r in rows], "s-", label="||J-I||/sqrt(d)")
    ax[2].set(title="how low-rank / far from identity", xlabel="layer")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"saved {path}")


# ------------------------------------------------------------------ loading (needs torch etc.)
def load_lens(path):
    if path.endswith(".npz"):
        z = np.load(path)
        J = {int(k[2:]): z[k].astype(np.float32) for k in z.files if k.startswith("J_")}
        return J, (z["W_U"].astype(np.float32) if "W_U" in z.files else None)
    import torch
    art = torch.load(path, map_location="cpu", weights_only=True)
    return {int(l): m.float().numpy() for l, m in art["J"].items()}, None


def load_unembedding(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    tok = AutoTokenizer.from_pretrained(model_id)
    del model
    return W, tok


# ------------------------------------------------------------------ self-test (synthetic)
def synthetic(seed=1, d=64, V=2000, n_layers=10, rank=6):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V, d)).astype(np.float32)
    for t in range(1500, V):  # planted near-duplicate tokens (like ' the' vs 'the')
        W[t] = W[(t - 1500) % 300] + 0.15 * rng.standard_normal(d)
    U = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    Vv = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    J = {}
    for l in range(n_layers):  # transport drifts toward the identity near the top
        scale = 3.0 * (1 - l / n_layers)
        J[l] = (np.eye(d) + scale * U @ np.diag(np.linspace(1, 0.4, rank)) @ Vv.T
                + 0.02 * rng.standard_normal((d, d)) / math.sqrt(d)).astype(np.float32)
    return J, W


def selftest():
    J, W = synthetic()
    with tempfile.TemporaryDirectory() as tmp:  # also exercises the .npz loader
        p = os.path.join(tmp, "lens.npz")
        np.savez(p, W_U=W, **{f"J_{l}": m for l, m in J.items()})
        J2, W2 = load_lens(p)
    ok = {"npz round trip": W2 is not None and np.array_equal(W, W2) and all(np.array_equal(J[l], J2[l]) for l in J)}
    rows, base, null_r90, ids = analyze(J2, W2, n_tokens=0, log=lambda *_: None)
    print_report(rows, base, null_r90, ids)
    grp = lambda t: int(t) if t < 300 else ((int(t) - 1500) % 300 if t >= 1500 else -int(t) - 1)
    _, i, j = max(rows, key=lambda r: r["mu"])["pairs"][0]
    dist = [r["dist_from_id"] for r in rows]
    ok["||J-I|| shrinks toward top layer"] = all(a >= b - 1e-9 for a, b in zip(dist, dist[1:]))
    ok["planted rank-6 structure found (r90/d < 0.25 every layer)"] = all(r["r90_frac"] < 0.25 for r in rows)
    ok["coherence above Gaussian null"] = all(r["mu"] > base["gaussian null"]["mu"] + 0.2 for r in rows)
    ok["certified k no larger than null"] = all(r["k_cert"] <= base["gaussian null"]["k_cert"] for r in rows)
    ok["top confusable pair is a planted duplicate"] = grp(ids[i]) == grp(ids[j]) and grp(ids[i]) >= 0
    ok["clusters worse conditioned than random subsets"] = all(r["clus_k2"] < r["rand_k2"] for r in rows)
    ok["no Welch violation"] = all(r["mu"] >= r["welch"] - 1e-9 for r in rows)
    print()
    for name, v in ok.items():
        print(f"[{'PASS' if v else 'FAIL'}] {name}")
    print("SELFTEST " + ("PASSED" if all(ok.values()) else "FAILED"))
    return all(ok.values())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list-lenses", action="store_true")
    ap.add_argument("--model", help="HF model id the lens was fitted on, e.g. gpt2")
    ap.add_argument("--lens", help="local lens artifact (.pt dict with key 'J', or .npz)")
    ap.add_argument("--hf-lens", help="filename inside neuronpedia/jacobian-lens")
    ap.add_argument("--n-tokens", type=int, default=8000)
    ap.add_argument("--full", action="store_true", help="use the whole vocabulary")
    ap.add_argument("--subset", choices=["random", "first"], default="random")
    ap.add_argument("--layers", type=int, nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="jcert_report")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.list_lenses:
        from huggingface_hub import list_repo_files
        for f in sorted(x for x in list_repo_files("neuronpedia/jacobian-lens") if x.endswith(".pt")):
            print(f)
        return
    if not a.model or not (a.lens or a.hf_lens):
        ap.error("give --model and one of --lens / --hf-lens (or run --selftest)")
    path = a.lens
    if a.hf_lens:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("neuronpedia/jacobian-lens", filename=a.hf_lens)
    J, W = load_lens(path)
    tok = None
    if W is None:
        W, tok = load_unembedding(a.model)
    d_lens = next(iter(J.values())).shape[0]
    if W.shape[1] != d_lens:
        sys.exit(f"d_model mismatch: lens has {d_lens}, model unembedding has {W.shape[1]}")
    rows, base, null_r90, ids = analyze(J, W, 0 if a.full else a.n_tokens, a.subset, a.seed, a.layers)
    print_report(rows, base, null_r90, ids, tok)
    write_csv(rows, base, a.out + ".csv")
    print(f"saved {a.out}.csv")
    if not a.no_plot:
        make_plot(rows, base, a.out + ".png")


if __name__ == "__main__":
    main()
