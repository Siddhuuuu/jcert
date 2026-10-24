#!/usr/bin/env python3
"""JCert rigor checks: stress the weak spots of the first test suite. numpy + scipy only.

R1  exact (no Monte Carlo) verification of the T4 moment identities
R2  T4 over many independent seeds, with standard errors
R3  T3': does it still hold when column errors come from a MATRIX error E (the real situation)?
R4  how matrix error is amplified into column error when J is ill-conditioned
R5  ideal problem vs algorithm: greedy nonnegative pursuit vs exact best support
"""
import itertools
import math
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jcert_real_lens import coherence_rows, unit_rows  # noqa: E402

SUMMARY = []


def report(name, ok, detail):
    if ok is not None:
        SUMMARY.append((name, ok))
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {name}\n        {detail}", flush=True)


# ---------------------------------------------------------------- R1
def r1_exact_moments():
    rng = np.random.default_rng(0)
    worst = 0.0
    for d in (3, 4, 5, 6):
        J = rng.standard_normal((d, d))
        U = np.array(list(itertools.product([-1.0, 1.0], repeat=d)))  # every probe, 2^d of them
        X = np.einsum("bi,bj->bij", U, U @ J)                          # u (u^T J)
        fro2 = (J ** 2).sum()
        worst = max(worst, np.abs(X.mean(0) - J).max(), abs((X ** 2).sum((1, 2)).mean() - d * fro2) / (d * fro2))
    d, N = 5, 2_000_000
    J = rng.standard_normal((d, d))
    u = rng.standard_normal((N, d))
    v = (u ** 2).sum(1) * ((u @ J) ** 2).sum(1)
    est, se, th = v.mean(), v.std() / math.sqrt(N), (d + 2) * (J ** 2).sum()
    ok = worst < 1e-12 and abs(est - th) < 4 * se
    report("R1 exact moments of probe estimator", ok,
           f"Rademacher: E[X]=J and E||X||_F^2 = d||J||_F^2 exactly over all 2^d probes (max error {worst:.1e}); "
           f"Gaussian: E||X||^2 = (d+2)||J||_F^2, Monte Carlo {est:.3f} +/- {se:.3f} vs theory {th:.3f}")


# ---------------------------------------------------------------- R2
def t4_ratio(d, n_exact, r, trials, rng):
    B = n_exact * d
    Jbar = rng.standard_normal((d, d)) / math.sqrt(d)
    M = (Jbar ** 2).sum()
    tau = math.sqrt(r * M) / d
    s2 = d * d * tau ** 2
    ee, er = [], []
    for _ in range(trials):
        Je = Jbar + tau * rng.standard_normal((n_exact, d, d))
        ee.append(((Je.mean(0) - Jbar) ** 2).sum())
        Jp = Jbar + tau * rng.standard_normal((B, d, d))
        U = rng.choice([-1.0, 1.0], size=(B, d))
        X = np.einsum("bi,bj->bij", U, np.einsum("bi,bij->bj", U, Jp))
        er.append(((X.mean(0) - Jbar) ** 2).sum())
    return (np.mean(er) / np.mean(ee)) / (1 + (d - 1) * M / (d * s2))


def r2_multiseed(seeds=8, trials=80):
    out, ok = [], True
    for r in (0.1, 1.0, 10.0):
        vals = np.array([t4_ratio(32, 20, r, trials, np.random.default_rng(100 + s)) for s in range(seeds)])
        m, se = vals.mean(), vals.std(ddof=1) / math.sqrt(seeds)
        ok &= abs(m - 1) <= max(3 * se, 0.01)
        out.append(f"r={r}: empirical/theory = {m:.4f} +/- {se:.4f} (SE over {seeds} seeds)")
    report("R2 T4 over independent seeds", ok, " | ".join(out))


# ---------------------------------------------------------------- R3 / R4
def r3_r4_matrix_noise():
    rng = np.random.default_rng(1)
    d, V = 256, 5000
    W = rng.standard_normal((V, d))
    Q1, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Q2, _ = np.linalg.qr(rng.standard_normal((d, d)))
    rows, ok = [], True
    for smin in (1.0, 0.3, 0.1, 0.03):
        s = np.logspace(0, math.log10(smin), d)
        J = (Q1 * s) @ Q2.T
        E = rng.standard_normal((d, d))
        E *= 0.05 * np.linalg.norm(J) / np.linalg.norm(E)           # 5% relative Frobenius error
        Lr, Le = W @ J, W @ (J + E)                                   # rows = lens vectors J^T w
        eps = np.linalg.norm(W @ E, axis=1) / np.linalg.norm(Lr, axis=1)
        mu = coherence_rows(unit_rows(Lr))[0]
        mu_hat = coherence_rows(unit_rows(Le))[0]
        em = eps.max()
        if em < 1:
            bound = math.cos(max(math.acos(mu) - 2 * math.asin(em), 0.0))
            ok &= mu_hat <= bound + 1e-9
            tail = f"mu {mu:.3f}->{mu_hat:.3f} <= T3' bound {bound:.3f}"
        else:
            tail = f"mu {mu:.3f}->{mu_hat:.3f}; T3' vacuous (eps_max>=1)"
        rows.append(f"sigma_min(J)={smin}: column error eps median {np.median(eps):.3f}, "
                    f"95% {np.quantile(eps, .95):.3f}, max {em:.3f}; {tail}")
    report("R3 T3' under correlated (matrix) noise  +  R4 error amplification", ok,
           " || ".join(rows) + "  (all with 5% relative Frobenius error in J)")


# ---------------------------------------------------------------- R5
def nn_omp(D, h, k):
    S, r = [], h.copy()
    for _ in range(k):
        c = D.T @ r
        c[S] = -np.inf
        S.append(int(np.argmax(c)))
        x, _ = nnls(D[:, S], h)
        r = h - D[:, S] @ x
    return tuple(sorted(S)), np.linalg.norm(r)


def r5_greedy_vs_exact(trials=300, d=10, V=14, noise=0.3):
    rng = np.random.default_rng(2)
    res = []
    for name in ("incoherent", "coherent (near-duplicate pairs)"):
        for k in (2, 3):
            hit, excess = 0, []
            for _ in range(trials):
                if name == "incoherent":
                    D = unit_rows(rng.standard_normal((V, d))).T
                else:
                    base = unit_rows(rng.standard_normal((V // 2, d)))
                    D = np.vstack([base, unit_rows(base + 0.3 * rng.standard_normal(base.shape))]).T
                S0 = rng.choice(V, k, replace=False)
                h = D[:, S0] @ rng.uniform(0.5, 2, k) + noise * rng.standard_normal(d)
                best = min((nnls(D[:, list(S)], h)[1], S) for S in itertools.combinations(range(V), k))
                Sg, rg = nn_omp(D, h, k)
                hit += Sg == tuple(sorted(best[1]))
                excess.append((rg - best[0]) / best[0])
            res.append(f"{name}, k={k}: greedy finds exact best support {hit / trials:.0%}, "
                       f"mean excess residual {np.mean(excess):.1%}")
    report("R5 greedy pursuit vs exact support (d=10, V=14, noise 0.3)", None, " | ".join(res))


if __name__ == "__main__":
    r1_exact_moments()
    r2_multiseed()
    r3_r4_matrix_noise()
    r5_greedy_vs_exact()
    n_ok = sum(ok for _, ok in SUMMARY)
    print(f"\n{n_ok}/{len(SUMMARY)} pass/fail checks passed (R5 is informational)")
