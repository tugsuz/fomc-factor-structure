"""Tools for panel models with multiple individual effects (17 Nov 2026).

Notation follows the lecture notes of S. C. Ahn "Panel models with multiple individual effects":
  r    number of individual effects used in estimation; r_o is the true number
  k    number of regressors (the meaning of the lecture notes; k is never a number of factors here)
  q    number of instruments; N units; T time periods
Data layout: Y is N x T. X is N x T x k, or None when there is no regressor. S is N x q (one row per unit).
All data are demeaned over the units inside the functions. This removes a common time effect, and it is the
same as a constant among the instruments. So r counts effects beyond the common time effect.

Part 1. Large N, small T: Ahn, Lee and Schmidt (2001, 2013).
  als_cu     continuous-updating GMM of the lecture notes (equation III) with H'H = I and A = second moment of the
             residuals. For a given beta the criterion is N times the sum of the smallest squared canonical
             correlations between the residuals and the instruments.
  als_gmm    two-step GMM with H = (I, -Psi)' and a weighting matrix that allows heteroskedasticity.
             The choice of the r periods in the normalization block is random (my choice, as in Ahn and Perez).
  als_count  J tests for r = 0, 1, ..., the sequential estimate and the BIC estimate of r_o.
Part 2. Large N, large T: Bai (2009), with the iteration of the lecture notes (p. 23).
  bai_ife    iterative principal components estimator and its standard errors.
Part 3. Data: stock panels from the TAQ window returns, with the traits of panel_traits.py.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2

import paths

TAQ = os.path.join(paths.ROOT, "data_wrds", "taq")
OUT = os.path.join(paths.RESULTS, "panel_effects_2026-11-17")
RISK = ["beta_mkt", "beta_y2", "beta_slope", "logcap", "vol"]
SECT = ["fin", "reit", "util", "energy", "tech", "health"]


# ---------------------------------------------------------------- Part 1: Ahn, Lee and Schmidt
def _c(A):
    return A - A.mean(0)


def cc2(U, S):
    """Squared canonical correlations of U (N x T) and S (N x q), largest first."""
    qu = np.linalg.qr(_c(U))[0]
    qs = np.linalg.qr(_c(S))[0]
    return np.linalg.svd(qu.T @ qs, compute_uv=False) ** 2


def fe_ols(Y, X):
    """Two-way within estimator (unit effects and time effects), the usual fixed effects estimator."""
    y = within2(Y)
    x = np.stack([within2(X[:, :, j]) for j in range(X.shape[2])], axis=2)
    return np.linalg.lstsq(x.reshape(-1, x.shape[2]), y.ravel(), rcond=None)[0]


def als_cu(Y, X, S, r):
    """Returns beta (or None), J and the degrees of freedom (T - r)(q - r) - k."""
    N, T = Y.shape
    q = S.shape[1]
    if X is None:
        return None, float(N * cc2(Y, S)[r:].sum()), (T - r) * (q - r)
    k = X.shape[2]
    f = lambda b: N * cc2(Y - X @ b, S)[r:].sum()
    b0 = fe_ols(Y, X)
    starts = [b0, np.zeros(k)]
    if k == 1:
        grid = b0[0] + np.linspace(-1.5, 1.5, 31)
        starts.append(np.array([grid[np.argmin([f(np.array([g])) for g in grid])]]))
    best = min((minimize(f, s, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8}) for s in starts), key=lambda o: o.fun)
    return best.x, float(best.fun), (T - r) * (q - r) - k


def als_gmm(Y, X, S, r, order=None, beta0=None):
    """Two-step GMM. Moments: E[s_i (x) H(psi)'(y_i - X_i beta)] = 0 with H = (I, -Psi)'.
    order: the last r periods of this order form the normalization block. Returns beta, J, df."""
    N, T = Y.shape
    q = S.shape[1]
    k = 0 if X is None else X.shape[2]
    order = np.arange(T) if order is None else np.asarray(order)
    a, b = order[:T - r], order[T - r:]
    Yc, Sc = _c(Y), _c(S)
    Xc = None if k == 0 else X - X.mean(0)

    def mom(par):
        U = Yc if k == 0 else Yc - Xc @ par[:k]
        E = U[:, a] if r == 0 else U[:, a] - U[:, b] @ par[k:].reshape(T - r, r).T
        return (E[:, :, None] * Sc[:, None, :]).reshape(N, -1)

    def root(W):
        val, vec = np.linalg.eigh(W)
        return vec * np.sqrt(np.clip(val, 0, None))

    p0 = np.zeros(k + (T - r) * r)
    if k:
        p0[:k] = fe_ols(Y, X) if beta0 is None else beta0
    if r:
        U = Yc if k == 0 else Yc - Xc @ p0[:k]
        fit = Sc @ np.linalg.lstsq(Sc, U[:, b], rcond=None)[0]
        p0[k:] = np.linalg.lstsq(fit, U[:, a], rcond=None)[0].T.ravel()
    C1 = root(np.kron(np.eye(T - r), np.linalg.inv(Sc.T @ Sc / N)))
    p1 = p0 if len(p0) == 0 else least_squares(lambda p: C1.T @ mom(p).mean(0), p0, method="lm").x
    W2 = np.linalg.pinv(np.cov(mom(p1).T, bias=True))
    C2 = root(W2)
    p2 = p1 if len(p1) == 0 else least_squares(lambda p: C2.T @ mom(p).mean(0), p1, method="lm").x
    g = mom(p2).mean(0)
    return (p2[:k] if k else None), float(N * g @ W2 @ g), (T - r) * (q - r) - k


def pick(J, df, N, level=0.05):
    """Three estimates of r_o from dicts J[r] and df[r]:
    seq   the first r that the J test does not reject at the given level;
    bic   the r that minimizes J - ln(N) * df (the BIC of the lecture notes with f(N) = ln N);
    bic2  the same with the lighter penalty 0.75 * ln(N) / ln(5) per degree of freedom. This is the "BIC2" of the
          working paper version of Ahn, Lee and Schmidt (2013), which did best in their simulations.
    If every r is rejected, the estimate is the next value (the cap). The criteria are 0 at the cap."""
    rs = sorted(J)
    cap = rs[-1] + 1
    seq = next((r for r in rs if chi2.sf(J[r], df[r]) >= level), cap)
    crit = {r: J[r] - np.log(N) * df[r] for r in rs}
    crit2 = {r: J[r] - 0.75 * np.log(N) / np.log(5) * df[r] for r in rs}
    crit[cap] = crit2[cap] = 0.0
    return seq, min(crit, key=crit.get), min(crit2, key=crit2.get)


def als_count(Y, X, S, method="cu", level=0.05, n_order=5, rng=None):
    """J, df and beta for each r with positive degrees of freedom; then the two estimates of r_o.
    method 'cu': the lecture notes' continuous-updating version (one answer, no normalization).
    method 'gmm': the two-step version, n_order random normalizations, the most frequent estimate."""
    N, T = Y.shape
    q = S.shape[1]
    k = 0 if X is None else X.shape[2]
    rs = [r for r in range(min(T, q)) if (T - r) * (q - r) - k > 0]
    if method == "cu":
        J, df, beta = {}, {}, {}
        for r in rs:
            beta[r], J[r], df[r] = als_cu(Y, X, S, r)
        seq, bic, bic2 = pick(J, df, N, level)
        return {"J": J, "df": df, "beta": beta, "seq": seq, "bic": bic, "bic2": bic2}
    rng = rng or np.random.default_rng(0)
    seqs, bics, bic2s, betas = [], [], [], []
    for _ in range(n_order):
        order = rng.permutation(T)
        J, df, beta = {}, {}, {}
        for r in rs:
            beta[r], J[r], df[r] = als_gmm(Y, X, S, r, order)
        s, b, b2 = pick(J, df, N, level)
        seqs.append(s)
        bics.append(b)
        bic2s.append(b2)
        betas.append(beta)
    mode = lambda v: max(set(v), key=v.count)
    return {"seq": mode(seqs), "bic": mode(bics), "bic2": mode(bic2s), "seqs": seqs, "bics": bics, "betas": betas}


# ---------------------------------------------------------------- Part 2: Bai (2009)
def within2(A):
    return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()


def bai_ife(Y, X, r, tol=1e-8, max_iter=3000):
    """Interactive effects estimator. Unit and time effects are removed first by two-way demeaning.
    Returns beta, standard errors as on the lecture notes (variance sigma_i^2 for unit i), standard errors
    clustered by time period (my addition: they allow correlation across units inside a period),
    the number of iterations, and for each regressor the share of its variance that is left after the
    factors and loadings are projected out. A share near 0 means that the coefficient is not identified
    (a regressor like trait_i * shock_t when the shock is one of the factors).
    r = 0 is the usual two-way fixed effects estimator."""
    N, T = Y.shape
    k = X.shape[2]
    y = within2(Y)
    x = np.stack([within2(X[:, :, j]) for j in range(k)], axis=2)
    beta = np.linalg.lstsq(x.reshape(-1, k), y.ravel(), rcond=None)[0]
    yt, xt, it = y, x, 0
    if r > 0:
        for it in range(1, max_iter + 1):
            U, s, Vt = np.linalg.svd(y - x @ beta, full_matrices=False)
            L, F = U[:, :r], Vt[:r].T
            pl = lambda A: A - L @ (L.T @ A)
            yt = pl(y) - (pl(y) @ F) @ F.T
            xt = np.stack([pl(x[:, :, j]) - (pl(x[:, :, j]) @ F) @ F.T for j in range(k)], axis=2)
            new = np.linalg.lstsq(xt.reshape(-1, k), yt.ravel(), rcond=None)[0]
            done = np.max(np.abs(new - beta)) < tol
            beta = new
            if done:
                break
    e = yt - xt @ beta
    bread = np.linalg.inv(np.einsum("itj,itl->jl", xt, xt))
    adj = N * T / max((N - r - 1) * (T - r - 1) - k, 1)
    m1 = np.einsum("itj,itl,i->jl", xt, xt, (e ** 2).mean(1))
    gt = np.einsum("itj,it->tj", xt, e)
    se1 = np.sqrt(np.diag(bread @ m1 @ bread) * adj)
    se2 = np.sqrt(np.diag(bread @ (gt.T @ gt) @ bread) * adj)
    left = (xt ** 2).sum((0, 1)) / (x ** 2).sum((0, 1))
    return beta, se1, se2, it, left


# ---------------------------------------------------------------- Part 3: data
def stock_windows():
    """Window returns of the stocks (100 * log). post = close - w30 is the return from 20 minutes after
    the event to 16:00. w15 runs from 5 minutes before to 10 minutes after. They share no price."""
    d = pd.read_csv(os.path.join(TAQ, "windows_all.csv.gz"),
                    usecols=["date", "set", "unscheduled", "kind", "permno", "w15", "w30", "close", "pre30"])
    d = d[d["kind"] == "stock"].dropna(subset=["permno"])
    d["permno"] = d["permno"].astype(int)
    d["post"] = d["close"] - d["w30"]
    return d


def day_sets(d):
    """Scheduled statement days in the data, and for each one its control day one week before (A) and
    one week after (B), when that day is in the data."""
    t = pd.read_csv(os.path.join(TAQ, "targets.csv"), dtype=str)
    have = set(d["date"].unique())
    st = sorted(x for x in t[(t["set"] == "statement") & (t["unscheduled"] == "0")]["date"] if x in have)
    c = t[(t["set"] == "control") & t["date"].isin(have)]
    A = {p: x for x, p in zip(c["date"], c["pair"]) if x < p}
    B = {p: x for x, p in zip(c["date"], c["pair"]) if x > p}
    return st, A, B


def panel(d, days, cols):
    """N x T arrays for the stocks that have every column on every day. Returns the permnos and a dict."""
    g = d[d["date"].isin(days)].dropna(subset=list(cols))
    cnt = g.groupby("permno")["date"].nunique()
    g = g[g["permno"].isin(cnt[cnt == len(days)].index)]
    return sorted(g["permno"].unique()), {c: g.pivot(index="permno", columns="date", values=c).reindex(columns=days).values for c in cols}


def traits(year):
    t = pd.read_csv(os.path.join(OUT, "traits.csv"))
    t = t[t["year"] == year].set_index("permno")
    for s in SECT:
        t[s] = (t["sector"] == s).astype(float)
    return t
