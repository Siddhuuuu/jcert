#!/usr/bin/env python3
"""JCert Gate 0 + pre-registered metrics (see PREREGISTRATION_v1.1.md, sections 3-4).

  python jcert_gate0.py --selftest   # numpy + scipy only: tests the metric code
  python jcert_gate0.py --run        # Kaggle GPU: Gate 0 on Gemma-2-2b (needs torch + transformer_lens)

The metric code is tested by --selftest. The --run path is written from the public TransformerLens
demo and nnsight tutorial and is UNTESTED against the real libraries (the authoring sandbox had no torch).
If it raises, paste the traceback; each gate is wrapped so one failure does not hide the others.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

# ------------------------------------------------------------------ metrics (numpy only)


def unit_rows(A):
    A = np.asarray(A, dtype=np.float64)
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def orth_basis(atoms, tol=1e-6):
    """Orthonormal basis (d x r) of the span of the rows of `atoms` (s x d)."""
    U, s, _ = np.linalg.svd(np.asarray(atoms, dtype=np.float64).T, full_matrices=False)
    keep = s > tol * s[0] if (s.size and s[0] > 0) else np.zeros(0, dtype=bool)
    return U[:, keep]


def component(h, atoms):
    """Orthogonal projection of h onto span(atoms): the J-space component."""
    Q = orth_basis(atoms)
    h = np.asarray(h, dtype=np.float64)
    return Q @ (Q.T @ h)


def E_metric(h, delta, atoms_S, atoms_S2):
    """Reselection-induced component displacement (pre-registered primary metric).
    E = 0 whenever the selected set is unchanged; E = 0 does NOT imply an unchanged set."""
    h = np.asarray(h, dtype=np.float64)
    hp = h + np.asarray(delta, dtype=np.float64)
    denom = np.linalg.norm(component(h, atoms_S))
    if denom < 1e-8 * np.linalg.norm(h):
        return float("nan")
    return float(np.linalg.norm(component(hp, atoms_S2) - component(hp, atoms_S)) / denom)


def E_extras(h, delta, atoms_S, atoms_S2):
    """Companion quantities reported next to E: absolute displacement over ||h|| and the baseline
    component fraction ||P_S h||^2 / ||h||^2 (small fractions inflate the relative metric)."""
    h = np.asarray(h, dtype=np.float64)
    hp = h + np.asarray(delta, dtype=np.float64)
    nh = np.linalg.norm(h)
    return {"E_abs": float(np.linalg.norm(component(hp, atoms_S2) - component(hp, atoms_S)) / nh),
            "comp_frac": float(np.linalg.norm(component(h, atoms_S)) ** 2 / nh ** 2)}


def cluster_key(token_str, morph=False):
    """Conservative equivalence class: strip space and the sentencepiece marker, lowercase.
    Tokens whose normalized form is empty form their own singleton class.
    morph=True additionally drops one trailing 's' when len > 3 (crude; sensitivity analysis only)."""
    s = token_str.replace("\u2581", " ").strip().lower()
    if s == "":
        return "<raw:%r>" % token_str
    if morph and len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def cluster_jaccard(ids_a, ids_b, decode, morph=False):
    return jaccard({cluster_key(decode(int(i)), morph) for i in ids_a},
                   {cluster_key(decode(int(i)), morph) for i in ids_b})


def log_sigma_min(atoms):
    """J1: log of the smallest singular value of the unit-normalized selected atoms."""
    return float(np.log(max(np.linalg.svd(unit_rows(atoms), compute_uv=False)[-1], 1e-300)))


def swap_margin(h, atoms_S, S_ids, corr_fn, get_atoms, pool=500):
    """J2: min over single swaps (remove j in S, add a from the `pool` atoms most correlated with the
    residual) of the increase in squared projection residual, divided by ||h||^2. Can be negative."""
    h = np.asarray(h, dtype=np.float64)
    A = unit_rows(atoms_S)
    Q = orth_basis(A)
    r = h - Q @ (Q.T @ h)
    base = float(r @ r)
    c = np.asarray(corr_fn(r), dtype=np.float64).copy()
    c[list(S_ids)] = -np.inf
    pool = min(pool, len(c) - len(S_ids))
    ids = np.argpartition(c, -pool)[-pool:]
    B = unit_rows(get_atoms(ids))
    best = np.inf
    for j in range(A.shape[0]):
        Qj = orth_basis(np.delete(A, j, axis=0))
        rj = h - Qj @ (Qj.T @ h)
        Bp = B - (B @ Qj) @ Qj.T
        den = (Bp ** 2).sum(1)
        ok = den > 1e-12
        gain = np.zeros(len(B))
        gain[ok] = (Bp[ok] @ rj) ** 2 / den[ok]
        best = min(best, float(rj @ rj) - float(gain.max()) - base)
    return best / float(h @ h)


def min_gap(h, S_ids, corr_fn, get_atoms):
    """B5: minimum over pursuit steps of (c1 - c2)/c1, the gap between the two best correlations
    (NNLS replay of the support order; approximate for gradient pursuit)."""
    from scipy.optimize import nnls

    h = np.asarray(h, dtype=np.float64)
    atoms = unit_rows(get_atoms(np.asarray(S_ids)))
    r, chosen, gaps = h.copy(), [], []
    for t in range(len(S_ids)):
        c = np.asarray(corr_fn(r), dtype=np.float64).copy()
        c[chosen] = -np.inf
        c2, c1 = np.partition(c, -2)[-2:]
        if c1 > 0:
            gaps.append((c1 - c2) / c1)
        chosen.append(int(S_ids[t]))
        x, _ = nnls(atoms[: t + 1].T, h)
        r = h - atoms[: t + 1].T @ x
    return float(min(gaps)) if gaps else float("nan")


def rho_ref(rho_real_values):
    """Pre-registered reference dose: clip(median measured relative error, 0.001, 0.03).
    Returns the unclipped median, the reference dose and their ratio (disclose when ratio != 1)."""
    med = float(statistics.median(rho_real_values))
    ref = float(min(max(med, 0.001), 0.03))
    return {"median": med, "ref": ref, "ratio_ref_over_median": ref / med if med > 0 else float("inf")}


def budget_plan(t_seconds, t_feature=0.0, prompts=150, positions=4, layers=5,
                n_primary=12000, n_dose=4500, cap_hours=5.0):
    """Pre-registered budget rule. Decompositions: 12,000 primary (3,000 units x (1 clean + 3 draws)) +
    4,500 dose-response (300 units x 5 doses x 3 draws). Feature cost: t_feature seconds per unit.
    If over cap: halve prompts once (min 60), then positions to 2, then layers to 2, else stop."""
    base_units = 150 * 4 * 5
    cfg = {"prompts": prompts, "positions": positions, "layers": layers}

    def hours(c):
        units = c["prompts"] * c["positions"] * c["layers"]
        scale = units / base_units
        return (scale * (n_primary + n_dose) * t_seconds + units * t_feature) / 3600.0

    for step in (None, ("prompts", 75), ("positions", 2), ("layers", 2)):
        if step:
            cfg[step[0]] = step[1]
        if hours(cfg) <= cap_hours:
            return {"status": "ok", "config": dict(cfg), "est_hours": round(hours(cfg), 2)}
    return {"status": "stop", "config": dict(cfg), "est_hours": round(hours(cfg), 2)}


def nnomp_ref(Dunit, h, k):
    """Reference nonnegative OMP (for tests only)."""
    from scipy.optimize import nnls

    S, r = [], np.asarray(h, dtype=np.float64).copy()
    for _ in range(k):
        c = Dunit @ r
        c[S] = -np.inf
        j = int(np.argmax(c))
        if c[j] <= 1e-12:
            break
        S.append(j)
        x, _ = nnls(Dunit[S].T, h)
        r = h - Dunit[S].T @ x
    return S


# ------------------------------------------------------------------ self-test


def selftest():
    rng = np.random.default_rng(0)
    res = {}
    d, V = 64, 3000
    D = unit_rows(rng.standard_normal((V, d)))
    corr = lambda r: D @ r
    get = lambda ids: D[np.asarray(ids)]
    h = D[rng.choice(V, 6, replace=False)].T @ rng.uniform(0.5, 2, 6) + 0.5 * rng.standard_normal(d)
    S = nnomp_ref(D, h, 25)
    A = D[S]
    c = component(h, A)
    res["projection: idempotent and residual orthogonal"] = (
        np.linalg.norm(component(c, A) - c) < 1e-9 and np.abs(A @ (h - c)).max() < 1e-9)
    res["E = 0 for same support in any order"] = E_metric(h, 0 * h, A, A[::-1]) < 1e-9
    delta = 0.3 * np.linalg.norm(h) * rng.standard_normal(d) / np.sqrt(d)
    S2 = nnomp_ref(D, h + delta, 25)
    e_val = E_metric(h, delta, A, D[S2])
    res["E finite and >= 0 for a different support"] = np.isfinite(e_val) and e_val >= 0 and S2 != S
    ex = E_extras(h, delta, A, D[S2])
    res["E_extras: E_abs = E * sqrt(comp_frac)"] = abs(ex["E_abs"] - e_val * np.sqrt(ex["comp_frac"])) < 1e-9
    # different supports with the same span must give E = 0 (E = 0 does not imply same support)
    A_alt = np.vstack([A[:-1], A[-1] + A[0]])
    res["E = 0 for a different support spanning the same subspace"] = E_metric(h, 0 * h, A, A_alt) < 1e-9
    e1, e2 = np.eye(4)[0:1], np.eye(4)[1:2]
    res["E is nan when the clean component is zero"] = np.isnan(E_metric(e2[0], 0 * e2[0], e1, e2))
    keys = [cluster_key(x) for x in ("\u2581the", "the", " The", "The")]
    res["conservative cluster rule merges case/space variants only"] = (
        len(set(keys)) == 1 and cluster_key("\u2581cats") == "cats" and cluster_key("class") == "class"
        and cluster_key("\u2581") != cluster_key(" ") and cluster_key("\u2581").startswith("<raw:"))
    res["morph option merges plurals (and over-merges 'class')"] = (
        cluster_key("cats", morph=True) == "cat" and cluster_key("class", morph=True) == "clas"
        and cluster_key("is", morph=True) == "is")
    dec = {1: "\u2581the", 2: "The", 3: "cats", 4: "cat"}.get
    res["cluster jaccard: conservative vs morph"] = (
        cluster_jaccard([1, 3], [2, 4], dec) == 1 / 3 and cluster_jaccard([1, 3], [2, 4], dec, morph=True) == 1.0)
    res["jaccard basics"] = abs(jaccard({1, 2, 3}, {2, 3, 4}) - 0.5) < 1e-12 and np.isnan(jaccard(set(), set()))
    res["sigma_min: orthonormal = 1, near-duplicate small"] = (
        abs(log_sigma_min(np.eye(5, 8))) < 1e-9
        and np.exp(log_sigma_min(np.array([[1, 0, 0], [1, 0.01, 0.0]]))) < 0.02)
    I = np.eye(8)
    Dsm = np.vstack([I, unit_rows(rng.standard_normal((32, 8)))])
    cs = lambda r: Dsm @ r
    gs = lambda ids: Dsm[np.asarray(ids)]
    m_neg = swap_margin(I[0].copy(), Dsm[[1, 2]], [1, 2], cs, gs, pool=10)
    m_pos = swap_margin(2 * I[0] + 3 * I[1], Dsm[[0, 1]], [0, 1], cs, gs, pool=10)
    res["swap margin: -1 when a perfect swap exists"] = abs(m_neg + 1.0) < 1e-9
    res["swap margin >= 0 when support is exact"] = np.isfinite(m_pos) and m_pos >= -1e-9
    g = min_gap(h, S[:8], corr, get)
    res["min_gap in (0, 1]"] = 0 < g <= 1
    r1, r2, r3 = rho_ref([0.0005, 0.0008, 0.0009]), rho_ref([0.02] * 3), rho_ref([0.2, 0.3, 0.1])
    res["rho_ref clips and reports the ratio"] = (
        r1["ref"] == 0.001 and abs(r1["ratio_ref_over_median"] - 1.25) < 1e-9
        and abs(r2["ref"] - 0.02) < 1e-12 and abs(r2["ratio_ref_over_median"] - 1) < 1e-9
        and r3["ref"] == 0.03 and abs(r3["ratio_ref_over_median"] - 0.15) < 1e-9)
    p1, p2, p3, p4 = (budget_plan(t) for t in (0.15, 1.5, 4.0, 20.0))
    res["budget rule: ok / halve prompts / fewer positions / stop"] = (
        p1["status"] == "ok" and p1["config"]["prompts"] == 150
        and p2["status"] == "ok" and p2["config"]["prompts"] == 75 and p2["config"]["positions"] == 4
        and p3["status"] == "ok" and p3["config"]["positions"] == 2
        and p4["status"] == "stop")
    pf = budget_plan(0.15, t_feature=8.0)
    res["budget rule counts feature cost"] = pf["status"] == "ok" and pf["config"]["prompts"] == 75
    for name, ok in res.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("SELFTEST " + ("PASSED" if all(res.values()) else "FAILED"))
    return all(res.values())


# ------------------------------------------------------------------ Gate 0 (needs torch + transformer_lens)

PROMPTS = (
    "Paris grew around islands in the Seine and became a center of art, politics, science, commerce, and education across Europe for many centuries.",
    "Astronomers compare planets by mass, orbit, atmosphere, temperature, moons, and composition when describing the structure of the Solar System.",
    "During photosynthesis, plants use light energy to transform water and carbon dioxide into chemical energy while releasing oxygen into the air.",
    "A computer program transforms input data through a sequence of instructions, stores intermediate state in memory, and returns an output to the user.",
    "The committee reviewed the proposal carefully, argued about the budget for several hours, and finally voted to approve a smaller version of the plan.",
)
POSITIONS = (-1, -5, -9)
# Published reference readout (nnsight Jacobian-lens tutorial, gemma-2-2b, bf16): layer -> J-lens top-1 token.
TUTORIAL_PROMPT = "Fact: The currency used in the country shaped like a boot is"
TUTORIAL_TOP1 = {4: " anything", 8: "\n\n\n", 13: "LookAnd", 17: "LookAnd", 21: " euro", 24: " the"}


def _collect(model, layers):
    hooks = {l: f"blocks.{l}.hook_out" for l in layers}
    acts = {}
    for p, text in enumerate(PROMPTS):
        _, cache = model.run_with_cache(model.to_tokens(text), names_filter=list(hooks.values()))
        for l, hn in hooks.items():
            for pos in POSITIONS:
                acts[(l, p, pos)] = cache[hn][0, pos].detach().float().cpu().numpy()
    return acts


def run_gate0(a):
    import gc

    import torch
    from huggingface_hub import hf_hub_download
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    out = {"args": vars(a), "gates": {}, "errors": {}}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out["device"] = dev

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    # G0.a load
    try:
        model = TransformerBridge.boot_transformers(a.model, device=dev, dtype=torch.float32)
        model.eval()
        try:
            lens = JacobianLens.from_pretrained(a.lens_name, model=model, revision=a.lens_revision)
        except TypeError:
            lens = JacobianLens.from_pretrained(a.lens_name, model=model)
            out["errors"]["G0.a note"] = "from_pretrained rejected revision=; loaded default revision"
        out["lens_metadata"] = str(getattr(lens, "metadata", None))
        out["gates"]["G0.a load"] = True
    except Exception as e:  # noqa: BLE001
        out["gates"]["G0.a load"] = False
        out["errors"]["G0.a"] = repr(e)
        return _finish(out, a)

    acts = _collect(model, a.layers)
    times, rel_err, recon_err, e0_ok, conv = [], [], [], True, set()
    t_sig, t_swap, t_gap = [], [], []
    try:
        for l in a.layers:
            Dt = lens.lens_vector_dictionary(model, l)
            D = (Dt.detach().float().cpu().numpy() if hasattr(Dt, "detach") else np.asarray(Dt)).astype(np.float32)
            corr_fn = lambda r, D=D: D @ np.asarray(r, dtype=np.float32)
            get_atoms = lambda ids, D=D: D[np.asarray(ids)]
            timed = False
            for (ll, p, pos), h_np in acts.items():
                if ll != l:
                    continue
                h = torch.tensor(h_np, device=dev)
                ts = []
                for _ in range(3):
                    sync()
                    t0 = time.perf_counter()
                    dec = lens.decompose(model, h, layer=l, k=a.k, algorithm="gradient_pursuit")
                    sync()
                    ts.append(time.perf_counter() - t0)
                times.append(statistics.median(ts[1:]))  # first call may build caches
                ids = dec.selected_support.detach().cpu().numpy()
                lib = dec.j_space_component.detach().float().cpu().numpy()
                rel_err.append(float(np.linalg.norm(component(h_np, D[ids]) - lib) / max(np.linalg.norm(lib), 1e-12)))
                sup = dec.support.detach().cpu().numpy()
                coords = dec.coordinates.detach().float().cpu().numpy()
                rec = dec.reconstruction.detach().float().cpu().numpy()
                errs = {"raw rows": np.linalg.norm(coords @ D[sup] - rec),
                        "unit rows": np.linalg.norm(coords @ unit_rows(D[sup]) - rec)}
                best = min(errs, key=errs.get)
                recon_err.append(float(errs[best] / max(np.linalg.norm(rec), 1e-12)))
                conv.add(best)
                e0_ok &= E_metric(h_np, 0 * h_np, D[ids], D[ids][::-1]) < 1e-9
                if not timed:  # time the proposed-score features once per layer (CPU upper bound)
                    t0 = time.perf_counter(); log_sigma_min(D[ids]); t_sig.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); swap_margin(h_np, D[ids], ids, corr_fn, get_atoms); t_swap.append(time.perf_counter() - t0)
                    t0 = time.perf_counter(); min_gap(h_np, ids, corr_fn, get_atoms); t_gap.append(time.perf_counter() - t0)
                    timed = True
            del D, Dt
            gc.collect()
        out["gates"]["G0.b metrics reproduce library"] = bool(max(rel_err) < 1e-3 and max(recon_err) < 1e-3 and e0_ok)
        out["max_component_rel_err"] = max(rel_err)
        out["max_reconstruction_rel_err"] = max(recon_err)
        out["reconstruction_convention"] = sorted(conv)
        out["t_gp_seconds_median"] = statistics.median(times)
        out["feature_seconds_per_unit"] = {"sigma": statistics.median(t_sig), "swap": statistics.median(t_swap),
                                           "gap": statistics.median(t_gap)}
        tf = sum(out["feature_seconds_per_unit"].values())
        if dev == "cuda":
            out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["gates"]["G0.c memory and timing measured"] = True
        out["budget_plan"] = budget_plan(out["t_gp_seconds_median"], t_feature=tf)
    except Exception as e:  # noqa: BLE001
        out["gates"]["G0.b metrics reproduce library"] = False
        out["gates"]["G0.c memory and timing measured"] = False
        out["errors"]["G0.b/c"] = repr(e)

    # G0.e non-degeneracy
    try:
        prof = lens.fraction_of_variance(model, list(PROMPTS), layers=list(a.layers), k=a.k, positions=list(POSITIONS))
        med = {int(l): float(prof.median[l]) for l in prof.layers}
        out["variance_fraction_median"] = med
        out["gates"]["G0.e non-degenerate (>=3% in >=3 layers)"] = sum(v >= 0.03 for v in med.values()) >= 3
    except Exception as e:  # noqa: BLE001
        out["gates"]["G0.e non-degenerate (>=3% in >=3 layers)"] = False
        out["errors"]["G0.e"] = repr(e)

    # G0.g coordinate compatibility with the published artifact and the tutorial's reference readout
    if not a.skip_compat:
        try:
            hf = getattr(model, "original_model", None)
            if hf is None:
                raise RuntimeError("cannot reach the underlying HF model via model.original_model")
            path = hf_hub_download("neuronpedia/jacobian-lens", filename=a.artifact_file, revision=a.artifact_revision)
            art = torch.load(path, map_location="cpu", weights_only=True)
            J = {int(k): v.float() for k, v in art["J"].items()}
            out["artifact"] = {"revision": a.artifact_revision, "n_prompts": art.get("n_prompts"), "d_model": art.get("d_model")}
            hdev = hf.lm_head.weight.device
            # (i) TL hook activation == HF hidden state
            worst_act = 0.0
            for p, text in enumerate(PROMPTS):
                with torch.no_grad():
                    hs = hf(input_ids=model.to_tokens(text).to(hdev), output_hidden_states=True).hidden_states
                for l in a.layers:
                    for pos in POSITIONS:
                        ref = hs[l + 1][0, pos].float().cpu().numpy()
                        worst_act = max(worst_act, float(np.linalg.norm(acts[(l, p, pos)] - ref) / np.linalg.norm(ref)))
            # (ii) library dictionary rows == artifact-derived rows W_U J (cosine)
            Wu = hf.lm_head.weight.detach().float().cpu()
            samp = np.random.default_rng(0).choice(Wu.shape[0], 200, replace=False)
            worst_cos = 1.0
            for l in a.layers:
                Dt = lens.lens_vector_dictionary(model, l)
                Dn = (Dt.detach().float().cpu().numpy() if hasattr(Dt, "detach") else np.asarray(Dt))[samp]
                ref = (Wu[samp] @ J[l]).numpy()
                cos = (Dn * ref).sum(1) / (np.linalg.norm(Dn, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
                worst_cos = min(worst_cos, float(cos.min()))
                del Dt, Dn
            # (iii) manual canonical readout reproduces the published top-1 tokens
            cap = hf.config.final_logit_softcapping
            toks = model.to_tokens(TUTORIAL_PROMPT).to(hdev)
            with torch.no_grad():
                hs = hf(input_ids=toks, output_hidden_states=True).hidden_states
                match = {}
                for l, want in TUTORIAL_TOP1.items():
                    est = hs[l + 1][0, -1].float() @ J[l].to(hdev).T
                    logits = hf.lm_head(hf.model.norm(est.to(hf.lm_head.weight.dtype))).float()
                    logits = cap * torch.tanh(logits / cap)
                    got = model.tokenizer.decode([int(logits.argmax())])
                    match[l] = (got, got == want)
            n_match = sum(v[1] for v in match.values())
            out["compat"] = {"worst_activation_rel_diff": worst_act, "worst_dictionary_cosine": worst_cos,
                             "tutorial_top1": {str(k): v for k, v in match.items()}}
            out["gates"]["G0.g compatible (act diff<1e-4, cos>=0.999, >=4/6 published top-1)"] = bool(
                worst_act < 1e-4 and worst_cos >= 0.999 and n_match >= 4)
        except Exception as e:  # noqa: BLE001
            out["gates"]["G0.g compatible (act diff<1e-4, cos>=0.999, >=4/6 published top-1)"] = False
            out["errors"]["G0.g"] = repr(e)

    # G0.d rho from a reduced-precision run
    del model, lens
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    out["gates"]["G0.d reference dose determined"] = False
    for name in ("bfloat16", "float16"):
        try:
            m2 = TransformerBridge.boot_transformers(a.model, device=dev, dtype=getattr(torch, name))
            m2.eval()
            lp = _collect(m2, a.layers)
            rhos = [float(np.linalg.norm(lp[k] - acts[k]) / np.linalg.norm(acts[k])) for k in acts]
            if not all(np.isfinite(rhos)):
                raise ValueError("non-finite activations in " + name)
            out["lowprec"] = name
            out["rho"] = rho_ref(rhos)
            out["gates"]["G0.d reference dose determined"] = True
            break
        except Exception as e:  # noqa: BLE001
            out["errors"]["G0.d " + name] = repr(e)
    return _finish(out, a)


def _finish(out, a):
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in out if k != "args"}, indent=2, default=str))
    print(f"\nsaved {a.out}. Gate 0 " + ("PASSED" if out["gates"] and all(out["gates"].values()) else "has failures: see gates/errors"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--lens-name", default="gemma-2-2b", help="identifier JacobianLens.from_pretrained expects")
    ap.add_argument("--lens-revision", default="a4114d7752d11eb546e6cf372213d7e75526d3a1")
    ap.add_argument("--artifact-file", default="gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt")
    ap.add_argument("--artifact-revision", default="a4114d7752d11eb546e6cf372213d7e75526d3a1")
    ap.add_argument("--skip-compat", action="store_true", help="skip G0.g (not recommended)")
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 13, 16, 19, 22])
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", default="gate0_report.json")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        run_gate0(a)
        return
    ap.error("give --selftest or --run")


if __name__ == "__main__":
    main()
