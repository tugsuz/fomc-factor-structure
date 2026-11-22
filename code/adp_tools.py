"""Ahn, Dieckmann and Perez (2018), "Is there a missing factor? A canonical correlation approach to factor
models", Review of Financial Economics 36(4), 321-347. Tools written from the paper (read in full, 10 Nov 2026).

Model (eq. 2): y_t = a + B1 f1_t + B2 f2_t + u_t, n response variables, m = all factors, r = factors
correlated with the k observed instruments z_t (f2 is the missing part). No missing factor means m = r.

1. m, the number of all factors (Section 2.4, eq. 12), by the GMM method of Ahn and Perez (2010):
   split y into group A (p variables) and group B (q variables), p + q = n. For a candidate mc, the first mc
   variables of group A are the normalizing block, and the moment conditions are
       E[(y_A2,t - G y_A1,t) (x) y_B,t] = 0,
   which is Phi_A' y_A,t (x) y_B,t with Phi_A' = [-G, I]. Hansen's J statistic is chi-square with
   df = (p - mc)(q - mc) if mc = m, and diverges if mc < m.
2. r, the number of factors correlated with z (eq. 13): the same with the whole y as group A and z as the
   instruments: E[(y_2,t - G y_1,t) (x) z_t] = 0, df = (n - rc)(k - rc).
3. Two ways to pick the number (Section 2.4): sequential testing at the 95 percent level, and the model
   selection criterion S(c) = J / ln(T) - (p - c)(q - c), the BIC(1) of Ahn and Perez (2010), minimised.
4. The estimate depends on the split and on the order inside group A (or inside y for r). So it is done
   for 100 random specifications and the final estimate is the most frequent value (footnote 8).
5. J uses the optimal weighting matrix with the Newey-West estimator, bandwidth 3 (footnote 7).
   GMM is two-step: first step with W = I (x) (Z'Z/T)^{-1} (2SLS equation by equation), second step with
   the inverse of the Newey-West matrix of the first-step moments.
6. CCA factors (Section 2.3): A = eigenvectors of S_yy^{-1} S_yz S_zz^{-1} S_zy for the r largest
   eigenvalues, normalized so that A' S_yy A = I; g_t = A' y_t.
7. RPC test (Section 2.1): regress each y_i on the regressors, take the first principal component of the
   residuals, add it, and see how much the adjusted R squared rises.

Choices the paper leaves open and that are made here (marked so they can be checked against Ahn and Perez 2010):
- Data are demeaned first (the paper omits the intercept "without loss of generality").
- S (the long-run variance of the moments) is computed from demeaned moments.
- Group sizes: p = ceil(n/2), q = n - p. The order inside group A is random.
- Sequential testing starts at 1 for m (as the paper writes) and at 0 for r. The criterion is minimised over
  0, 1, ..., max for both, where max = min(p, q) - 1 for m and min(n, k) - 1 for r.
"""
import numpy as np
from scipy import stats


def nw(G, L):
    """Newey-West long-run variance of the T x M moment series G (demeaned), Bartlett weights, bandwidth L."""
    G = G - G.mean(0, keepdims=True)
    T = len(G)
    S = G.T @ G / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        C = G[l:].T @ G[:-l] / T
        S += w * (C + C.T)
    return S


def gmm_j(Y2, Y1, Z, L=3):
    """J statistic for E[(Y2_t - G Y1_t) (x) Z_t] = 0. Y2: T x J, Y1: T x c (c may be 0), Z: T x q."""
    T, J = Y2.shape
    q = Z.shape[1]
    c = Y1.shape[1]

    def moments(theta):
        Gm = theta.reshape(J, c) if c else np.zeros((J, 0))
        E = Y2 - Y1 @ Gm.T  # T x J
        return np.einsum("tj,tq->tjq", E, Z).reshape(T, J * q)

    if c == 0:
        theta = np.zeros(0)
    else:
        s = (Z.T @ Y2 / T).T.reshape(J * q)  # stacked by equation j, each block q
        Qb = Z.T @ Y1 / T  # q x c
        Q = np.kron(np.eye(J), Qb)  # Jq x Jc
        W1 = np.kron(np.eye(J), np.linalg.inv(Z.T @ Z / T))
        theta = np.linalg.solve(Q.T @ W1 @ Q, Q.T @ W1 @ s)
    S = nw(moments(theta), L)
    W2 = np.linalg.inv(S)
    if c:
        theta = np.linalg.solve(Q.T @ W2 @ Q, Q.T @ W2 @ s)
    gbar = moments(theta).mean(0)
    Jstat = float(T * gbar @ W2 @ gbar)
    df = J * q - J * c
    return Jstat, df, float(stats.chi2.sf(Jstat, df))


def _pick(Js, dfs, ps, cands, T, start, alpha):
    seq = None
    for c in cands:
        if c < start:
            continue
        if ps[c] >= alpha:
            seq = c
            break
    if seq is None:
        seq = max(cands) + 1  # every candidate rejected
    crit = {c: Js[c] / np.log(T) - dfs[c] for c in cands}
    bic = min(crit, key=crit.get)
    return seq, bic


def estimate_m(Y, n_spec=100, L=3, alpha=0.05, rng=None):
    """Ahn-Perez GMM estimate of the number of all factors, over random group specifications."""
    rng = np.random.default_rng(0) if rng is None else rng
    Y = Y - Y.mean(0, keepdims=True)
    T, n = Y.shape
    p = int(np.ceil(n / 2))
    q = n - p
    cands = list(range(0, min(p, q)))
    seqs, bics = [], []
    for _ in range(n_spec):
        perm = rng.permutation(n)
        A, Bg = Y[:, perm[:p]], Y[:, perm[p:]]
        Js, dfs, ps = {}, {}, {}
        for c in cands:
            Js[c], dfs[c], ps[c] = gmm_j(A[:, c:], A[:, :c], Bg, L)
        s, b = _pick(Js, dfs, ps, cands, T, 1, alpha)
        seqs.append(s)
        bics.append(b)
    return np.array(seqs), np.array(bics), (p, q)


def estimate_r(Y, Z, n_spec=100, L=3, alpha=0.05, rng=None):
    """Ahn-Perez GMM estimate of the number of factors correlated with the instruments Z, over random orders of Y."""
    rng = np.random.default_rng(0) if rng is None else rng
    Y = Y - Y.mean(0, keepdims=True)
    Z = Z - Z.mean(0, keepdims=True)
    T, n = Y.shape
    k = Z.shape[1]
    cands = list(range(0, min(n, k)))
    seqs, bics = [], []
    for _ in range(n_spec):
        perm = rng.permutation(n)
        Yp = Y[:, perm]
        Js, dfs, ps = {}, {}, {}
        for c in cands:
            Js[c], dfs[c], ps[c] = gmm_j(Yp[:, c:], Yp[:, :c], Z, L)
        s, b = _pick(Js, dfs, ps, cands, T, 0, alpha)
        seqs.append(s)
        bics.append(b)
    return np.array(seqs), np.array(bics)


def mode(a):
    v, c = np.unique(a, return_counts=True)
    return int(v[np.argmax(c)]), {int(x): round(float(y) / len(a), 2) for x, y in zip(v, c)}


def cca_factors(Y, Z, r):
    """CCA factors g = A'y (Section 2.3) and the canonical correlations."""
    Y = Y - Y.mean(0, keepdims=True)
    Z = Z - Z.mean(0, keepdims=True)
    T = len(Y)
    Syy, Syz, Szz = Y.T @ Y / T, Y.T @ Z / T, Z.T @ Z / T
    ev, V = np.linalg.eigh(Syy)
    Sm = V @ np.diag(ev ** -0.5) @ V.T  # S_yy^{-1/2}
    Mx = Sm @ Syz @ np.linalg.solve(Szz, Syz.T) @ Sm
    lam, U = np.linalg.eigh(Mx)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    A = Sm @ U[:, :r]  # A' S_yy A = I
    return Y @ A, np.sqrt(np.clip(lam[: min(Y.shape[1], Z.shape[1])], 0, None)), A


def adj_r2(Y, X):
    """Adjusted R squared of each column of Y on (1, X)."""
    T = len(Y)
    Xc = np.column_stack([np.ones(T), X])
    E = Y - Xc @ np.linalg.lstsq(Xc, Y, rcond=None)[0]
    r2 = 1 - (E ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    kx = Xc.shape[1] - 1
    return 1 - (1 - r2) * (T - 1) / (T - kx - 1), E


def rpc(Y, X):
    """RPC test: adjusted R squared with X, then with X plus the first principal component of the residuals."""
    a0, E = adj_r2(Y, X)
    E = E - E.mean(0)
    U, S, Vt = np.linalg.svd(E, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    a1, _ = adj_r2(Y, np.column_stack([X, pc1]))
    return a0, a1
