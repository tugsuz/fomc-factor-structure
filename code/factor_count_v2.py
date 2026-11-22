"""Estimators for the number of factors. Corrected copy of factor_count.py (10 Nov 2026).

Source of truth: Ahn and Horenstein (2013), Econometrica 81(3), 1203-1227, read from the published article. Page numbers below are the journal's.

What the paper defines, and where this file does it (the paper writes k for a candidate number of
factors; the project writes r for a number of factors, so the outputs are r_er and r_gr):
  mu_k    k-th largest eigenvalue of X X'/(N T); m = min(N, T)              p. 1205   eigenvalues()
  V(k)    sum of mu_j for j > k                                             p. 1207   ahn_horenstein()
  ER(k)   mu_k / mu_{k+1},  k = 1..kmax                                     p. 1207
  GR(k)   ln[V(k-1)/V(k)] / ln[V(k)/V(k+1)]
          = ln(1 + mu*_k) / ln(1 + mu*_{k+1}),  mu*_k = mu_k / V(k)         p. 1207
  mock    mu_0 = V(0)/ln(m); with it k = 0 is allowed (Corollary 1, eq. 4)  p. 1208
  kmax    kmax1 = 2 rmax if a maximum rmax is known;
          kmax2 = min(kmax*, 0.1 m), kmax* = #{k >= 1: mu_k >= V(0)/m}      p. 1208   kmax2()
  demean  double demeaning x_it - mean_t - mean_i + grand mean, recommended
          even without time or individual effects                           p. 1209   double_demean()

What changed against factor_count.py (the estimator itself was already right):
  1. kmax is capped by the numerical rank, not only by m - 2. After demeaning the rank is at
     most m - 1, so the last eigenvalue is zero and V(k + 1) can be zero for a large kmax.
     With kmax = 8 this never binds.
  2. The mock eigenvalue takes a multiple (mock_multiple, default 1). The paper says the
     corollary holds for any multiple and that finite-sample results depend on it (p. 1208).
     "switch" in the output is the multiple at which the ER count moves between 0 and the
     best k >= 1:  switch = max_k ER(k) * ln(m) * mu_1 / V(0).  The count is 0 when the
     multiple is above switch. So switch far from 1 means the 0-or-not decision is robust.
  3. The output has the shares mu_k / V(0), kmax2, and both sequences from k = 0.
  4. Benchmarks used in the paper's simulations are added: Bai and Ng (2002) BIC3 and IC_p2,
     and the edge distribution (ED) estimator of Onatski (2010). The ED steps were checked
     against a secondary description (arXiv 2101.10383, section 3.2.2), not against the
     original article. Check them there before quoting ED numbers in the paper.

Two points that are easy to get wrong:
  - Double demeaning does not remove a factor. It replaces the loadings L by L minus their
    cross-sectional mean (p. 1209, eq. 6 and 7). A factor leaves the panel only if its
    loadings are the same for every series (a time effect). The market stays in a panel of
    stocks and bond ETFs after double demeaning, because their betas differ.
  - Footnote 3 (p. 1209): the time effect is a factor with constant loadings. If the
    cross-section mean explains the series well, it should be used as an estimated factor.
    So the total count is the count on demeaned data plus one in that case.

Notation: X is T x N (rows are periods or event days, columns are series). r is a number of
factors. Plain numpy.
"""
import numpy as np


def double_demean(X):
    """x_it minus the series mean, minus the period mean, plus the grand mean (p. 1209)."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=0, keepdims=True) - X.mean(axis=1, keepdims=True) + X.mean()


def eigenvalues(X):
    """Eigenvalues of X X'/(N T), largest first, length m = min(N, T) (p. 1205)."""
    T, N = X.shape
    S = X @ X.T if T <= N else X.T @ X
    ev = np.linalg.eigvalsh(S / (N * T))[::-1]
    return np.clip(ev, 0.0, None)


def kmax2(mu, m):
    """kmax2 = min(kmax*, 0.1 m), kmax* = #{k >= 1: mu_k >= V(0)/m} (p. 1208). At least 1."""
    kstar = int(np.sum(mu >= mu.sum() / m))
    return max(1, min(kstar, int(0.1 * m)))


def ahn_horenstein(X, kmax=8, demean=True, allow_zero=True, mock_multiple=1.0):
    """ER and GR estimates of the number of factors (Theorem 1 and Corollary 1).

    kmax: an integer, or "kmax2" for the paper's data-based rule.
    demean: True applies double demeaning first (the paper's recommendation).
    Returns a dict: r_er, r_gr, k (the grid), er, gr, eigenvalues, shares, kmax, kmax2,
    switch, N, T, m.
    """
    X = np.asarray(X, dtype=float)
    if demean:
        X = double_demean(X)
    T, N = X.shape
    m = min(N, T)
    mu = eigenvalues(X)
    V0 = mu.sum()
    k2 = kmax2(mu, m)
    if kmax == "kmax2" or kmax is None:
        kmax = k2
    rank = int(np.sum(mu > 1e-12 * mu[0]))
    kmax = int(max(1, min(kmax, rank - 2)))

    V = np.concatenate([np.cumsum(mu[::-1])[::-1], [0.0]])        # V[k] = sum_{j > k} mu_j, k = 0..m
    mu_full = np.concatenate([[mock_multiple * V0 / np.log(m)], mu])   # mu_full[k] = mu_k, k = 0..m
    k0 = 0 if allow_zero else 1
    ks = np.arange(k0, kmax + 1)
    er = mu_full[ks] / mu_full[ks + 1]
    mu_star = mu_full / np.where(V > 0, V, np.nan)                # mu*_k = mu_k / V(k)
    gr = np.log1p(mu_star[ks]) / np.log1p(mu_star[ks + 1])
    er_pos = mu[:kmax] / mu[1:kmax + 1]                           # ER(1..kmax)
    switch = float(er_pos.max() * np.log(m) * mu[0] / V0)
    return {"r_er": int(ks[np.nanargmax(er)]), "r_gr": int(ks[np.nanargmax(gr)]), "k": ks, "er": er, "gr": gr,
            "eigenvalues": mu, "shares": mu / V0, "kmax": kmax, "kmax2": k2, "switch": switch,
            "N": N, "T": T, "m": m}


def bai_ng(X, kmax=8, demean=True):
    """Bai and Ng (2002): IC_p2 and BIC3. Returns (r_icp2, r_bic3).

    IC_p2(k) = ln V(k) + k (N + T)/(N T) ln(min(N, T)).
    BIC3(k)  = V(k) + k s2 (N + T - k) ln(N T)/(N T), with s2 = V(kmax).
    """
    X = np.asarray(X, dtype=float)
    if demean:
        X = double_demean(X)
    T, N = X.shape
    mu = eigenvalues(X)
    V = np.concatenate([[mu.sum()], mu.sum() - np.cumsum(mu)])
    ks = np.arange(0, kmax + 1)
    ic = np.log(V[ks]) + ks * (N + T) / (N * T) * np.log(min(N, T))
    bic3 = V[ks] + ks * V[kmax] * (N + T - ks) * np.log(N * T) / (N * T)
    return int(ks[np.argmin(ic)]), int(ks[np.argmin(bic3)])


def onatski_ed(X, rmax=8, demean=True, max_iter=20):
    """Edge distribution estimator of Onatski (2010). See point 4 of the header.

    Start with j = rmax + 1. Regress mu_j, ..., mu_{j+4} on a constant and
    (j-1)^(2/3), ..., (j+3)^(2/3); delta = 2 |slope|. The estimate is the largest i <= rmax
    with mu_i - mu_{i+1} >= delta (0 if there is none). Set j = estimate + 1 and repeat
    until the estimate does not change.
    """
    X = np.asarray(X, dtype=float)
    if demean:
        X = double_demean(X)
    mu = eigenvalues(X)
    j, r = rmax + 1, None
    for _ in range(max_iter):
        y = mu[j - 1:j + 4]
        x = np.arange(j - 1, j + 4) ** (2.0 / 3.0)
        slope = np.polyfit(x, y, 1)[0]
        delta = 2 * abs(slope)
        gaps = mu[:rmax] - mu[1:rmax + 1]
        idx = np.where(gaps >= delta)[0]
        r_new = int(idx.max() + 1) if len(idx) else 0
        if r_new == r:
            break
        r, j = r_new, r_new + 1
    return r
