"""Estimators for the number of factors in a large panel.

Ahn and Horenstein (2013, Econometrica 81(3)): eigenvalue ratio (ER) and
growth ratio (GR), with the mock eigenvalue that allows zero factors
(their Corollary 1) and double demeaning.
Bai and Ng (2002, Econometrica 70(1)): IC_p2, as a benchmark.

Notation follows the paper. X is T x N (rows are days, columns are assets).
mu_k is the k-th largest eigenvalue of X X' / (N T). V(k) = sum_{j>k} mu_j is
the mean squared residual after k principal components.

Plain numpy.
"""
import numpy as np


def double_demean(X):
    """Subtract column means and row means, add back the grand mean."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=0, keepdims=True) - X.mean(axis=1, keepdims=True) + X.mean()


def eigenvalues(X):
    """Eigenvalues of X X' / (N T), largest first, length min(N, T)."""
    T, N = X.shape
    if T <= N:
        S = X @ X.T
    else:
        S = X.T @ X
    ev = np.linalg.eigvalsh(S / (N * T))[::-1]
    return np.clip(ev, 0.0, None)


def default_kmax(mu, m):
    """kmax2 of Ahn-Horenstein: min(#{k: mu_k >= V(0)/m}, 0.1 m), at least 1."""
    V0 = mu.sum()
    kstar = int(np.sum(mu >= V0 / m))
    return max(1, min(kstar, int(0.1 * m)))


def ahn_horenstein(X, kmax=None, demean=True, allow_zero=True):
    """ER and GR estimates of the number of factors.

    Returns a dict with k_er, k_gr, the ER and GR sequences (index k),
    the eigenvalues and kmax. With allow_zero, k runs from 0 and the mock
    eigenvalue mu_0 = V(0) / ln(m) is used (Corollary 1).
    """
    X = np.asarray(X, dtype=float)
    if demean:
        X = double_demean(X)
    T, N = X.shape
    m = min(N, T)
    mu = eigenvalues(X)
    if kmax is None:
        kmax = default_kmax(mu, m)
    kmax = int(min(kmax, m - 2))

    # V(k) for k = 0..m: V[k] = sum_{j>k} mu_j  (mu is 1-indexed in the paper)
    tail = np.concatenate([np.cumsum(mu[::-1])[::-1], [0.0]])  # tail[i] = sum_{j>=i} mu (0-indexed)
    V = tail  # V[k] = sum of mu_{k+1}, ..., mu_m  in 1-indexed terms

    mu_full = np.concatenate([[V[0] / np.log(m)], mu])  # mu_full[k] = mu_k, k = 0..m
    k0 = 0 if allow_zero else 1
    ks = np.arange(k0, kmax + 1)

    er = mu_full[ks] / mu_full[ks + 1]

    # GR(k) = ln(1 + mu*_k) / ln(1 + mu*_{k+1}),  mu*_k = mu_k / V(k)
    def mu_star(k):
        return mu_full[k] / V[k]

    gr = np.log1p(mu_star(ks)) / np.log1p(mu_star(ks + 1))

    return {
        "k_er": int(ks[np.argmax(er)]),
        "k_gr": int(ks[np.argmax(gr)]),
        "k": ks,
        "er": er,
        "gr": gr,
        "eigenvalues": mu,
        "kmax": kmax,
        "N": N,
        "T": T,
    }


def bai_ng_icp2(X, kmax=8, demean=True):
    """Bai-Ng (2002) IC_p2: ln V(k) + k (N+T)/(N T) ln min(N, T)."""
    X = np.asarray(X, dtype=float)
    if demean:
        X = double_demean(X)
    T, N = X.shape
    mu = eigenvalues(X)
    V = np.concatenate([[mu.sum()], mu.sum() - np.cumsum(mu)])
    ks = np.arange(0, kmax + 1)
    ic = np.log(V[ks]) + ks * (N + T) / (N * T) * np.log(min(N, T))
    return int(ks[np.argmin(ic)]), ic
