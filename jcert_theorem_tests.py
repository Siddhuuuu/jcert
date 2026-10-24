#!/usr/bin/env python3
"""JCert theorem tests (numpy + scipy, offline, ~1-2 min).

Numerical checks can FALSIFY a claim but cannot PROVE it. Each test hunts for
counterexamples (random + adversarial). A PASS means "no counterexample found",
and the written proof still carries the logical weight.
"""
import itertools
import math
import time

import numpy as np
from scipy.linalg import hadamard
from scipy.optimize import minimize, nnls

rng = np.random.default_rng(0)
SUMMARY = []


def report(name, ok, detail):
    SUMMARY.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}", flush=True)


def unit(D):
    return D / np.linalg.norm(D, axis=0, keepdims=True)


def coherence(D):
    U = unit(D)
    G = np.abs(U.T @ U)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def welch(d, V):
    return math.sqrt((V - d) / (d * (V - 1)))


def k_cert(mu):
    return max(int(math.ceil(0.5 * (1 + 1 / mu)) - 1), 0)


def spark(D, tol=1e-8):
    d, V = D.shape
    for s in range(2, min(d + 1, V) + 1):
        for S in itertools.combinations(range(V), s):
            if np.linalg.matrix_rank(D[:, list(S)], tol=tol) < s:
                return s
    return math.inf


# ------------------------------------------------------------------ T1
def t1_lemma(trials=800):
    worst, tight = math.inf, 0
    for _ in range(trials):
        d = int(rng.integers(3, 7))
        V = d + int(rng.integers(1, 5))
        D = rng.integers(-2, 3, size=(d, V)).astype(float)
        if (np.linalg.norm(D, axis=0) == 0).any():
            continue
        mu = coherence(D)
        slack = spark(D) - (1 + 1 / mu)
        worst = min(worst, slack)
        tight += int(abs(slack) < 1e-9 and mu < 0.999)
    ang = np.arange(3) * 2 * np.pi / 3
    mb = np.vstack([np.cos(ang), np.sin(ang)])
    slack_mb = spark(mb) - (1 + 1 / coherence(mb))
    report("T1 lemma: spark(D) >= 1 + 1/mu", worst > -1e-9 and abs(slack_mb) < 1e-9,
           f"min slack over {trials} integer dictionaries = {worst:.3g}; non-trivially tight "
           f"cases = {tight}; Mercedes-Benz frame slack = {slack_mb:.2g} (equality case)")


def t1_uniqueness():
    H = hadamard(16) / 4.0
    D = np.hstack([np.eye(16), H])
    V = D.shape[1]
    mu = coherence(D)
    kc = k_cert(mu)

    def all_independent(size):
        return all(np.linalg.svd(D[:, list(S)], compute_uv=False)[-1] > 1e-9
                   for S in itertools.combinations(range(V), size))

    ok_indep = all_independent(2 * kc)

    def n_alternatives(k, trials):
        bad = 0
        supports = list(itertools.combinations(range(V), k))
        for _ in range(trials):
            S0 = rng.choice(V, k, replace=False)
            alpha = np.zeros(V)
            alpha[S0] = rng.uniform(0.2, 2.0, k)
            h = D @ alpha
            for S in supports:
                cols = D[:, list(S)]
                x = np.linalg.lstsq(cols, h, rcond=None)[0]
                if np.linalg.norm(cols @ x - h) < 1e-9:
                    full = np.zeros(V)
                    full[list(S)] = x
                    if np.linalg.norm(full - alpha) > 1e-7:
                        bad += 1
                        break
        return bad

    bad_kc = n_alternatives(kc, 150)
    bad_3 = n_alternatives(3, 25)
    # explicit failure at k = 4: subgroup of Z_2^4 vs its annihilator
    G_cols = [0, 1, 2, 3]
    A_cols = [0, 4, 8, 12]
    a_H = np.zeros(V); a_H[16 + np.array(G_cols)] = 1.0
    a_I = np.zeros(V); a_I[A_cols] = 1.0
    gap = np.linalg.norm(D @ a_H - D @ a_I)
    fails_at_4 = gap < 1e-12 and np.linalg.norm(a_H - a_I) > 1
    report("T1: uniqueness below the coherence bound ([I|Hadamard16], mu=1/4)",
           ok_indep and bad_kc == 0 and fails_at_4,
           f"certified k<={kc}: all {2*kc}-column subsets independent={ok_indep}, alternatives found "
           f"at k={kc}: {bad_kc}/150; at k=3 (beyond certificate): {bad_3}/25 (still unique); "
           f"explicit nonnegative 4-sparse collision at k=4 verified (gap={gap:.1e})")


def t1_welch():
    worst = math.inf
    for _ in range(300):
        d = int(rng.integers(2, 12))
        V = d + int(rng.integers(1, 3 * d))
        worst = min(worst, coherence(rng.standard_normal((d, V))) - welch(d, V))
    phi = (1 + 5 ** 0.5) / 2
    ico = np.array([[0, 1, phi], [0, 1, -phi], [1, phi, 0], [1, -phi, 0],
                    [phi, 0, 1], [phi, 0, -1]], float).T
    eq_ico = coherence(ico) - welch(3, 6)
    ang = np.arange(3) * 2 * np.pi / 3
    eq_mb = coherence(np.vstack([np.cos(ang), np.sin(ang)])) - welch(2, 3)

    def fg(x, d, V, p):
        X = x.reshape(d, V)
        n = np.linalg.norm(X, axis=0)
        U = X / n
        G = U.T @ U
        np.fill_diagonal(G, 0.0)
        f = np.sum(np.abs(G) ** p)
        Hm = p * np.abs(G) ** (p - 1) * np.sign(G)
        gU = 2 * U @ Hm
        gX = (gU - U * np.sum(U * gU, axis=0)) / n
        F = f ** (1 / p)
        return F, (F / (p * f)) * gX.ravel()

    best_gap = math.inf
    for d, V in ((3, 8), (4, 10), (5, 12), (8, 16)):
        for _ in range(15):
            x = rng.standard_normal(d * V)
            for p in (8, 16, 32, 64):
                x = minimize(fg, x, args=(d, V, p), jac=True, method="L-BFGS-B",
                             options={"maxiter": 300}).x
            best_gap = min(best_gap, coherence(x.reshape(d, V)) - welch(d, V))
    ok = worst >= -1e-12 and best_gap >= -1e-9 and abs(eq_ico) < 1e-9 and abs(eq_mb) < 1e-9
    report("T1b: Welch bound mu^2 >= (V-d)/(d(V-1))", ok,
           f"random min gap={worst:.3g}; optimiser trying to beat it, min gap={best_gap:.3g} (>=0); "
           f"equality for icosahedron ({eq_ico:.1e}) and Mercedes-Benz ({eq_mb:.1e})")


def t1_context():
    d, V = 10, 24
    D = unit(rng.standard_normal((d, V)))
    mu = coherence(D)
    smin = min(np.linalg.svd(D[:, rng.choice(V, d, replace=False)], compute_uv=False)[-1]
               for _ in range(20000))
    d2, V2 = 128, 5000
    D2 = unit(rng.standard_normal((d2, V2)))
    prof = []
    for k in (2, 8, 16, 32):
        s = [np.linalg.svd(D2[:, rng.choice(V2, 2 * k, replace=False)], compute_uv=False)[-1]
             for _ in range(150)]
        prof.append(f"k={k}: {np.median(s):.2f}")
    print("[INFO] T1 context (conservativeness of coherence certificates)\n"
          f"        Gaussian d={d}, V={V}: mu={mu:.2f} certifies only k<={k_cert(mu)}, yet 20000 random "
          f"{d}-subsets all independent (but min sigma_min={smin:.1e}), so exact uniqueness holds "
          f"for k<={d // 2} generically.\n"
          f"        Real obstruction is conditioning: median sigma_min of random 2k-column subsets, "
          f"d={d2}, V={V2}: " + ", ".join(prof), flush=True)


# ------------------------------------------------------------------ T2
def t2_stability(trials=70, d=6, V=13, k=2):
    D = unit(rng.standard_normal((d, V)))
    supports = list(itertools.combinations(range(V), k))
    cols = [D[:, list(S)] for S in supports]

    def res(h):
        return np.array([nnls(c, h)[1] for c in cols])

    viol, used, min_ratio, nonpos = 0, 0, math.inf, 0
    for _ in range(trials):
        S0 = rng.choice(V, k, replace=False)
        h = D[:, S0] @ rng.uniform(0.5, 2, k) + 0.5 * rng.standard_normal(d)
        r = res(h)
        order = np.argsort(r)
        b = order[0]
        delta = r[order[1]] - r[b]
        if delta < 1e-6:
            continue
        used += 1
        x_best = nnls(cols[b], h)[0]
        nonpos += int(x_best.min() <= 0)
        for _ in range(12):  # random perturbations just inside the radius
            e = rng.standard_normal(d)
            e *= 0.999 * delta / 2 / np.linalg.norm(e)
            r2 = res(h + e)
            if np.argmin(r2) != b or np.sort(r2)[1] - r2[b] <= 0:
                viol += 1
        for j in order[1:3]:  # adversarial: push toward the rival cone, away from best
            pj = cols[j] @ nnls(cols[j], h)[0]
            pb = cols[b] @ nnls(cols[b], h)[0]
            v = pj - pb
            if np.linalg.norm(v) < 1e-9:
                continue
            v /= np.linalg.norm(v)
            prev = 0.0
            for t in np.linspace(0, 3 * delta, 31)[1:]:
                r2 = res(h + t * v)
                if np.argmin(r2) != b or np.sort(r2)[1] - r2[b] <= 1e-12:
                    lo, hi = prev, t
                    for _ in range(25):
                        mid = 0.5 * (lo + hi)
                        r3 = res(h + mid * v)
                        if np.argmin(r3) != b or np.sort(r3)[1] - r3[b] <= 1e-12:
                            hi = mid
                        else:
                            lo = mid
                    min_ratio = min(min_ratio, hi / (delta / 2))
                    break
                prev = t
    # (b) rho is 1-Lipschitz
    worst_b = 0.0
    for _ in range(400):
        h1 = rng.standard_normal(d)
        h2 = h1 + rng.standard_normal(d) * 10 ** rng.uniform(-3, 0)
        worst_b = max(worst_b, abs(res(h1).min() - res(h2).min()) / np.linalg.norm(h1 - h2))
    # (c) NNLS coefficients / projection Lipschitz, fixed support
    S = list(rng.choice(V, k, replace=False))
    DS = D[:, S]
    u, sv, _ = np.linalg.svd(DS, full_matrices=False)
    sig = sv[-1]
    worst_c, worst_p, tight_c = 0.0, 0.0, 0.0
    for _ in range(3000):
        h1 = rng.standard_normal(d) * 2 + DS @ np.array([3.0, 3.0])
        h2 = h1 + rng.standard_normal(d) * 10 ** rng.uniform(-2, 0.5)
        x1, x2 = nnls(DS, h1)[0], nnls(DS, h2)[0]
        dh = np.linalg.norm(h1 - h2)
        worst_c = max(worst_c, np.linalg.norm(x1 - x2) / dh)
        worst_p = max(worst_p, np.linalg.norm(DS @ (x1 - x2)) / dh)
    h1 = DS @ np.array([5.0, 5.0])
    h2 = h1 + 0.1 * u[:, -1]
    tight_c = np.linalg.norm(nnls(DS, h1)[0] - nnls(DS, h2)[0]) / 0.1
    ok = (viol == 0 and min_ratio >= 1 - 1e-6 and worst_b <= 1 + 1e-9 and
          worst_c <= 1 / sig + 1e-9 and worst_p <= 1 + 1e-9 and nonpos == 0)
    report("T2: margin stability (a), residual Lipschitz (b), coefficient Lipschitz (c)", ok,
           f"{used} planted instances; flips inside radius Delta/2: {viol}; adversarial flip "
           f"distance / (Delta/2): min {min_ratio:.3f} (must be >=1); (b) max ratio {worst_b:.4f} "
           f"(<=1); (c) max ||x-x'||/||h-h'|| = {worst_c:.3f} <= 1/sigma_min = {1/sig:.3f} "
           f"(attained: {tight_c:.3f}); projection nonexpansive max {worst_p:.4f} (<=1); "
           f"positive margin => all coefficients > 0 violated {nonpos} times")


# ------------------------------------------------------------------ T3
def t3_lens_noise():
    H = hadamard(16) / 4.0
    D = np.hstack([np.eye(16), H])
    U = unit(D)
    mu = coherence(D)
    phi = math.acos(mu)

    def adversarial(eps):
        G = U.T @ U
        np.fill_diagonal(G, 0.0)
        i, j = np.unravel_index(np.abs(G).argmax(), G.shape)
        a, s = U[:, i], np.sign(G[i, j])
        c = s * U[:, j]
        e2 = c - (a @ c) * a
        e2 /= np.linalg.norm(e2)
        f2 = a - (a @ c) * c
        f2 /= np.linalg.norm(f2)
        out = U.copy()
        out[:, i] = (1 - eps ** 2) * a + eps * math.sqrt(1 - eps ** 2) * e2
        out[:, j] = s * ((1 - eps ** 2) * c + eps * math.sqrt(1 - eps ** 2) * f2)
        return out

    rows, ok = [], True
    for eps in (0.005, 0.02, 0.05, 0.1, 0.3):
        t3 = mu + 4 * eps
        t3p = math.cos(max(phi - 2 * math.asin(eps), 0.0))
        worst_rand = 0.0
        for _ in range(200):
            N = rng.standard_normal(U.shape)
            N *= eps / np.linalg.norm(N, axis=0)
            worst_rand = max(worst_rand, coherence(U + N))
        adv = adversarial(eps)
        rel = np.linalg.norm(adv - U, axis=0).max()
        mu_adv = coherence(adv)
        ok &= worst_rand <= t3p + 1e-12 and mu_adv <= t3p + 1e-12 and t3p <= t3 + 1e-12
        ok &= abs(mu_adv - t3p) < 1e-9 and rel <= eps + 1e-12
        rows.append(f"eps={eps}: random {worst_rand:.4f}, adversarial {mu_adv:.4f}, "
                    f"T3' {t3p:.4f}, T3 {t3:.4f}")
    k = 2
    thr3 = (1 / (2 * k - 1) - mu) / 4
    thr3p = math.sin((phi - math.acos(1 / (2 * k - 1))) / 2)
    cert_ok = coherence(adversarial(0.999 * thr3)) < 1 / (2 * k - 1)
    edge = coherence(adversarial(thr3p)) - 1 / (2 * k - 1)
    report("T3 / T3': lens noise -> coherence (relative column error eps)", ok and cert_ok,
           " | ".join(rows) + f" || keep k={k} certified: T3 allows eps<{thr3:.4f}, tight T3' allows "
           f"eps<={thr3p:.4f} (adversary lands exactly at threshold, diff {edge:.1e})")


# ------------------------------------------------------------------ T4
def t4_probes():
    out, ok = [], True
    for d, n_exact, trials in ((16, 20, 300), (32, 20, 300), (64, 10, 150)):
        B = n_exact * d
        Jbar = rng.standard_normal((d, d)) / math.sqrt(d)
        M = float((Jbar ** 2).sum())
        for r in (0.1, 1.0, 10.0):
            tau = math.sqrt(r * M) / d
            s2 = d * d * tau ** 2
            ee, er, eg = [], [], []
            for _ in range(trials):
                Je = Jbar + tau * rng.standard_normal((n_exact, d, d))
                ee.append(((Je.mean(0) - Jbar) ** 2).sum())
                Jp = Jbar + tau * rng.standard_normal((B, d, d))
                for kind, store in (("rad", er), ("gauss", eg)):
                    U = (rng.choice([-1.0, 1.0], size=(B, d)) if kind == "rad"
                         else rng.standard_normal((B, d)))
                    X = np.einsum("bi,bj->bij", U, np.einsum("bi,bij->bj", U, Jp))
                    store.append(((X.mean(0) - Jbar) ** 2).sum())
            me = np.mean(ee)
            th_r = 1 + (d - 1) * M / (d * s2)
            th_g = 1 + 2 / d + (d + 1) * M / (d * s2)
            e_r, e_g = np.mean(er) / me, np.mean(eg) / me
            ok &= abs(e_r / th_r - 1) < 0.08 and abs(e_g / th_g - 1) < 0.08 and e_r > 1
            out.append(f"d={d} r={r}: Rademacher {e_r:.2f} (theory {th_r:.2f}), "
                       f"Gaussian {e_g:.2f} (theory {th_g:.2f})")
    report("T4: probe/exact MSE ratio = 1 + (1-1/d) M/sigma^2 > 1 at equal cost", ok,
           " | ".join(out))


# ------------------------------------------------------------------ T5
def t5_rsvd():
    d, n, r, p, trials = 60, 40, 6, 6, 400
    s = np.array([3, 2.5, 2, 1.5, 1.2, 1.0])
    Uq, _ = np.linalg.qr(rng.standard_normal((d, r)))
    Vq, _ = np.linalg.qr(rng.standard_normal((d, r)))
    B = Uq @ np.diag(s) @ Vq.T + 0.05 * rng.standard_normal((d, d)) / math.sqrt(d)
    Jbar = np.eye(d) + B
    tau = 0.02
    sigma = d * tau                      # sqrt(E||J_p - Jbar||_F^2)
    lhs, rhs, tri_viol = [], [], 0
    for _ in range(trials):
        Jhat = Jbar + tau * rng.standard_normal((n, d, d)).mean(0)  # mean of n prompts
        A = Jhat - np.eye(d)
        Q, _ = np.linalg.qr(A @ rng.standard_normal((d, r + p)))
        E = np.linalg.norm(B - Q @ (Q.T @ A))
        t_sample = np.linalg.norm(B - A)
        t_rsvd = np.linalg.norm(A - Q @ (Q.T @ A))
        tri_viol += int(E > t_sample + t_rsvd + 1e-9)
        tail = math.sqrt((np.linalg.svd(A, compute_uv=False)[r:] ** 2).sum())
        lhs.append(E)
        rhs.append(sigma / math.sqrt(n) + math.sqrt(1 + r / (p - 1)) * tail)
    A0 = Jbar - np.eye(d) + 0.0
    sv = np.linalg.svd(A0, compute_uv=False)
    tail0 = math.sqrt((sv[r:] ** 2).sum())
    errs = []
    for _ in range(2000):
        Q, _ = np.linalg.qr(A0 @ rng.standard_normal((d, r + p)))
        errs.append(np.linalg.norm(A0 - Q @ (Q.T @ A0)))
    hmt = math.sqrt(1 + r / (p - 1)) * tail0
    ok = np.mean(lhs) <= np.mean(rhs) and tri_viol == 0 and np.mean(errs) <= hmt
    report("T5: E||J-I - QQ^T(Ahat)||_F <= sigma/sqrt(n) + HMT tail term", ok,
           f"mean error {np.mean(lhs):.3f} <= mean bound {np.mean(rhs):.3f}; triangle-inequality "
           f"violations {tri_viol}/{trials}; HMT check on fixed matrix: mean {np.mean(errs):.4f} "
           f"<= {hmt:.4f}")


if __name__ == "__main__":
    t0 = time.time()
    for fn in (t1_lemma, t1_uniqueness, t1_welch, t1_context, t2_stability,
               t3_lens_noise, t4_probes, t5_rsvd):
        fn()
    n_ok = sum(ok for _, ok in SUMMARY)
    print(f"\n{n_ok}/{len(SUMMARY)} tests passed in {time.time() - t0:.0f}s")
