"""JCert day-1 sanity checks. numpy only, runs offline in about a minute.

Check A: how large a sparsity k can coherence alone certify (Welch bound + random dictionary)?
Check B: random-probe vs exact estimator of the mean Jacobian at equal backward-pass budget.
Rule of the project: every claimed result gets a numerical test like these before you trust it.
"""
import numpy as np

# ---------------- Check A: coherence certificates ----------------
def welch(d, V):
    """Lower bound on mutual coherence of any V unit vectors in R^d."""
    return np.sqrt((V - d) / (d * (V - 1)))

def k_cert(mu):
    """Largest integer k with k < 0.5*(1 + 1/mu): sparse representation is then unique."""
    return int(np.ceil(0.5 * (1 + 1 / mu)) - 1)

def max_coherence(D, block=2000):
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    V, m = D.shape[1], 0.0
    for i in range(0, V, block):
        G = np.abs(D[:, i:i + block].T @ D)
        for j in range(G.shape[0]):
            G[j, i + j] = 0.0  # drop self-correlation
        m = max(m, G.max())
    return m

def check_a(V=50257):
    print(f"A1. Best-case certified k (Welch bound), vocab V={V}")
    for d in (512, 768, 1024, 2048, 4096):
        mu = welch(d, V)
        print(f"  d={d:5d}  mu_min={mu:.4f}  certified k <= {k_cert(mu)}")
    d = 100
    while k_cert(welch(d, V)) < 25:
        d += 1
    print(f"  smallest d where even a perfect dictionary certifies k=25: d={d}")
    rng = np.random.default_rng(0)
    d2, V2 = 768, 10000
    mu = max_coherence(rng.standard_normal((d2, V2)))
    print(f"A2. Random Gaussian dictionary d={d2}, V={V2}: mu={mu:.4f} "
          f"(Welch {welch(d2, V2):.4f}) -> certified k <= {k_cert(mu)}")

# ---------------- Check B: probes vs exact at equal budget ----------------
def check_b(d=32, n_exact=20, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    B = n_exact * d  # exact needs d backward passes per prompt; a probe costs 1
    Jbar = rng.standard_normal((d, d)) / np.sqrt(d)
    M = float((Jbar ** 2).sum())
    print(f"B. d={d}, budget B={B} backward passes ({n_exact} prompts if exact)")
    for r in (0.1, 1.0, 10.0):  # r = prompt variance / ||Jbar||^2
        tau = np.sqrt(r * M) / d
        s2 = d * d * tau ** 2
        err_e, err_p = [], []
        for _ in range(trials):
            Je = Jbar + tau * rng.standard_normal((n_exact, d, d))
            err_e.append(((Je.mean(0) - Jbar) ** 2).sum())
            Jp = Jbar + tau * rng.standard_normal((B, d, d))
            U = rng.choice([-1.0, 1.0], size=(B, d))
            X = np.einsum("bi,bj->bij", U, np.einsum("bi,bij->bj", U, Jp))
            err_p.append(((X.mean(0) - Jbar) ** 2).sum())
        emp = np.mean(err_p) / np.mean(err_e)
        theory = 1 + (d - 1) * M / (d * s2)
        print(f"  r={r:5.1f}  MSE ratio probe/exact: empirical {emp:6.2f}, theory {theory:6.2f}")

if __name__ == "__main__":
    check_a()
    check_b()
