"""GMM estimator of the number of factors (Ahn and Perez 2010), 11 Nov 2026.

Written from the working paper version (Perez and Ahn 2007, MPRA Paper 4862) and from Ahn, Dieckmann and
Perez (2018). This is a new file. adp_tools.py is not changed. What is different from adp_tools.py:
  1. Weighting. bandwidth 0 (White) is the default. The working paper recommends it for the criterion method.
     Event windows are weeks apart, so serial correlation is not the main worry. bandwidth 3 is still possible.
  2. The BIC criterion J - ln(T) * df is also evaluated at the largest value (full rank), where it is 0.
  3. The sequential test can use any level. level_T(T) gives one reading of the working paper's rule
     (a level that falls with T and equals 0.05 at T = 500).
  4. rule25() applies the working paper's rule for a weak factor to the frequency table of the random splits.
Notation: m = number of all factors, r = number of factors correlated with observed variables z.
"""
import numpy as np
from scipy.stats import chi2


def _hac(G, L):
    G = G - G.mean(0)
    T = len(G)
    S = G.T @ G / T
    for l in range(1, L + 1):
        C = G[l:].T @ G[:-l] / T
        S += (1 - l / (L + 1)) * (C + C.T)
    return S


def gmm_j(Y2, Y1, Z, L=0):
    """Two-step GMM J test of E[z (y2 - a - B'y1)] = 0.
    The data are demeaned first. This is the same as an intercept in each equation and a constant instrument.
    Y1 is None when there is no regressor (rank 0). Returns J, degrees of freedom, p-value."""
    T = len(Z)
    Z = Z - Z.mean(0)
    Y2 = Y2 - Y2.mean(0)
    n2, q = Y2.shape[1], Z.shape[1]
    c = 0 if Y1 is None else Y1.shape[1]
    df = n2 * (q - c)
    if n2 * q > T / 2:                       # too many moments for this T: the weighting matrix is not reliable
        return np.nan, df, np.nan
    m0 = (Y2[:, :, None] * Z[:, None, :]).reshape(T, n2 * q).mean(0)
    if c == 0:
        E, D = Y2, None
    else:
        Y1 = Y1 - Y1.mean(0)
        fit = Z @ np.linalg.solve(Z.T @ Z, Z.T @ Y1)
        B = np.linalg.lstsq(fit, Y2, rcond=None)[0]          # first step: 2SLS, equation by equation
        E = Y2 - Y1 @ B
        D = np.kron(np.eye(n2), Z.T @ Y1 / T)
    Si = np.linalg.pinv(_hac((E[:, :, None] * Z[:, None, :]).reshape(T, n2 * q), L))
    m = m0 if c == 0 else m0 - D @ np.linalg.solve(D.T @ Si @ D, D.T @ Si @ m0)
    J = float(T * m @ Si @ m)
    return J, df, float(chi2.sf(J, df))


def level_T(T):
    """One reading of the working paper's rule: 0.05 at T = 500, larger for smaller T (capped at 0.20)."""
    return min(0.20, 0.05 * np.sqrt(500.0 / T))


def _select(J, df, T, level, start):
    cands = [c for c in sorted(J) if not np.isnan(J[c])]
    if not cands:
        return np.nan, np.nan
    cap = max(J) + 1
    seq = cap
    for c in cands:
        if c >= start and chi2.sf(J[c], df[c]) >= level:
            seq = c
            break
    crit = {c: J[c] - np.log(T) * df[c] for c in cands}
    crit[cap] = 0.0
    return seq, min(crit, key=crit.get)


def count_m(X, n_spec=100, L=0, level=0.05, rng=None, start=1):
    """Number of all factors in X (T x N), no instruments. Each specification splits the series at random
    into group g (P = ceil(N/2) series) and instruments (Q = N - P series, at most 10).
    Returns the sequential and the BIC estimates of the n_spec specifications, and (P, Q)."""
    rng = rng or np.random.default_rng(0)
    T, N = X.shape
    Q = min(N // 2, 10)
    P = N - Q
    seq, bic = [], []
    for _ in range(n_spec):
        perm = rng.permutation(N)
        g, z = X[:, perm[:P]], X[:, perm[P:]]
        J, df = {}, {}
        for c in range(min(P, Q)):
            J[c], df[c], _ = gmm_j(g[:, c:], g[:, :c] if c else None, z, L)
        s, b = _select(J, df, T, level, start)
        seq.append(s)
        bic.append(b)
    return np.array(seq, float), np.array(bic, float), (P, Q)


def count_r(Y, Z, n_spec=100, L=0, level=0.05, rng=None):
    """Number of factors in Y (T x n) that are correlated with the observed variables Z (T x q).
    All series of Y are the responses and Z are the instruments. The order of Y is random in each specification.
    The largest possible value is min(n, q). If the estimate equals it, every testable value was rejected."""
    rng = rng or np.random.default_rng(0)
    T, n = Y.shape
    q = Z.shape[1]
    seq, bic = [], []
    for _ in range(n_spec):
        y = Y[:, rng.permutation(n)]
        J, df = {}, {}
        for c in range(min(n, q)):
            J[c], df[c], _ = gmm_j(y[:, c:], y[:, :c] if c else None, Z, L)
        s, b = _select(J, df, T, level, 0)
        seq.append(s)
        bic.append(b)
    return np.array(seq, float), np.array(bic, float)


def table(a):
    """Frequency table {value: share}, the most frequent value first."""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {}
    v, n = np.unique(a, return_counts=True)
    o = np.argsort(-n, kind="stable")
    return {int(v[i]): round(float(n[i]) / len(a), 2) for i in o}


def rule25(a):
    """Most frequent value, and the value after the working paper's rule: if the second most frequent value
    has a share of 25 percent or more and is larger than the most frequent one, take the second one."""
    t = list(table(a).items())
    if not t:
        return None, None
    mode = t[0][0]
    if len(t) > 1 and t[1][1] >= 0.25 and t[1][0] > mode:
        return mode, t[1][0]
    return mode, mode


def cca(Y, Z):
    """Canonical correlations of Y and Z (both demeaned inside), largest first."""
    Y = Y - Y.mean(0)
    Z = Z - Z.mean(0)
    qy = np.linalg.qr(Y)[0]
    qz = np.linalg.qr(Z)[0]
    return np.linalg.svd(qy.T @ qz, compute_uv=False)
