"""GMM estimator of the number of factors, version 2 (17 Nov 2026).

This version follows the published paper: Ahn and Perez (2010), Journal of Empirical Finance 17(4), 783-802,
Section 3.2 (pages 787-788), Section 5.1 (page 790) and Appendix B (pages 800-801).
ap_tools.py is not changed. It was written from the working paper of 2007.

Notation: L is the number of factors used in a test (the paper's letter). m = number of all factors,
r = number of factors correlated with observed variables. k is the number of regressors in the panel lecture notes
and is not used in this file. k is never a number of factors.

What changed against ap_tools.py, and where the paper says it:
  1. The sequential test starts at L = 0 (Appendix B, step S1). ap_tools.py started at 1 for m.
  2. The test level is 0.05 * sqrt(500 / T) (page 790). ap_tools.py had the same rule with a cap of 0.20.
  3. The criteria are MS(L) = J(L) / f(T) - g(L), minimized over L = 0, ..., Q - 1 (page 788, Appendix B):
       BIC1  f = ln(T)     g = (P - L)(Q - L)
       BIC2  f = log(T)    g = (P - L)(Q - L)      (log is read as the logarithm with base 10)
       BIC3  f = ln(T)     g = (P - L)(Q + 1)
       AIC   f = 1         g = 2 (P - L)(Q - L)
     ap_tools.py had BIC1 only. It also allowed the value Q, with criterion 0. The paper does not. v2 follows the paper.
  4. The rule for a weak factor (page 795) is stated for BIC2: if the second most frequent value has 25 percent or
     more of the random splits, look at it. rule25() is unchanged. Use it on the BIC2 estimates.
  5. The weighting matrix is the White matrix, the plain second moment of the moment functions (not demeaned).
     ap_tools.py demeaned them. centered=True gives the old version.
Unchanged, and in line with the paper: the intercept and the constant instrument (by demeaning the data),
two groups of about equal size, 100 random splits, the most frequent estimate, and for candidate factors
all responses in group one and the candidate factors in group two.
"""
import numpy as np
from scipy.stats import chi2

from ap_tools import cca, rule25, table  # unchanged helpers


def gmm_j(Y2, Y1, Z, L=0, centered=False):
    """Two-step GMM J test of E[z (y2 - a - B'y1)] = 0. Returns J, degrees of freedom and the number of moments.
    L is the Newey and West bandwidth here (0 = White)."""
    T = len(Z)
    Z = Z - Z.mean(0)
    Y2 = Y2 - Y2.mean(0)
    n2, q = Y2.shape[1], Z.shape[1]
    c = 0 if Y1 is None else Y1.shape[1]
    if n2 * q > T / 2:
        return np.nan, n2 * (q - c)
    m0 = (Y2[:, :, None] * Z[:, None, :]).reshape(T, n2 * q).mean(0)
    if c == 0:
        E, D = Y2, None
    else:
        Y1 = Y1 - Y1.mean(0)
        fit = Z @ np.linalg.solve(Z.T @ Z, Z.T @ Y1)
        E = Y2 - Y1 @ np.linalg.lstsq(fit, Y2, rcond=None)[0]
        D = np.kron(np.eye(n2), Z.T @ Y1 / T)
    G = (E[:, :, None] * Z[:, None, :]).reshape(T, n2 * q)
    if centered:
        G = G - G.mean(0)
    S = G.T @ G / T
    for l in range(1, L + 1):
        C = G[l:].T @ G[:-l] / T
        S += (1 - l / (L + 1)) * (C + C.T)
    Si = np.linalg.pinv(S)
    m = m0 if c == 0 else m0 - D @ np.linalg.solve(D.T @ Si @ D, D.T @ Si @ m0)
    return float(T * m @ Si @ m), n2 * (q - c)


def select(J, P, Q, T, level=None):
    """Estimates from the J statistics J[L], L = 0, ..., Q - 1. Returns a dict with seq, bic1, bic2, bic3, aic."""
    Ls = [l for l in sorted(J) if not np.isnan(J[l])]
    if not Ls:
        return {c: np.nan for c in ["seq", "bic1", "bic2", "bic3", "aic"]}
    level = 0.05 * np.sqrt(500.0 / T) if level is None else level
    df = {l: (P - l) * (Q - l) for l in Ls}
    seq = next((l for l in Ls if chi2.sf(J[l], df[l]) >= level), Ls[-1] + 1)   # every value rejected: one more
    arg = lambda f, g: min(Ls, key=lambda l: J[l] / f - g(l))
    return {"seq": seq, "bic1": arg(np.log(T), lambda l: df[l]), "bic2": arg(np.log10(T), lambda l: df[l]),
            "bic3": arg(np.log(T), lambda l: (P - l) * (Q + 1)), "aic": arg(1.0, lambda l: 2 * df[l])}


def count_m(X, n_spec=100, L=0, rng=None, level=None, centered=False):
    """Number of all factors in X (T x N). Random split into group one (P) and group two (Q = N // 2, at most 10).
    The first P - l series of group one are the dependent variables, the last l are the regressors (Appendix B).
    Returns a dict of arrays, one estimate for each split and criterion."""
    rng = rng or np.random.default_rng(0)
    T, N = X.shape
    Q = min(N // 2, 10)
    P = N - Q
    out = {c: [] for c in ["seq", "bic1", "bic2", "bic3", "aic"]}
    for _ in range(n_spec):
        perm = rng.permutation(N)
        g, z = X[:, perm[:P]], X[:, perm[P:]]
        J = {l: gmm_j(g[:, :P - l], g[:, P - l:] if l else None, z, L, centered)[0] for l in range(Q)}
        for c, v in select(J, P, Q, T, level).items():
            out[c].append(v)
    return {c: np.array(v, float) for c, v in out.items()}


def count_r(Y, Z, n_spec=100, L=0, rng=None, level=None, centered=False):
    """Number of factors in Y (T x n) that are correlated with the candidate factors Z (T x q)."""
    rng = rng or np.random.default_rng(0)
    T, P = Y.shape
    Q = Z.shape[1]
    out = {c: [] for c in ["seq", "bic1", "bic2", "bic3", "aic"]}
    for _ in range(n_spec):
        y = Y[:, rng.permutation(P)]
        J = {l: gmm_j(y[:, :P - l], y[:, P - l:] if l else None, Z, L, centered)[0] for l in range(min(P, Q))}
        for c, v in select(J, P, Q, T, level).items():
            out[c].append(v)
    return {c: np.array(v, float) for c, v in out.items()}
