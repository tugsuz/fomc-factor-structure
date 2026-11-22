"""Panel tools, version 2 (17 Nov 2026): checked against the published papers. panel_tools.py is not changed.

Notation as in the lecture notes of S. C. Ahn: r effects in estimation, r_o the true number, k regressors (k is never a
number of factors), q instruments, N units, T periods. The papers write p for r.

A. Ahn, Lee and Schmidt (2013), Journal of Econometrics 174, 1-14.
  pick_v2       the criteria as published on page 8, and the plain BIC of page 6:
                  AIC   J - 2 df
                  BIC   J - ln(N) df                         (page 6 with a = 1)
                  BIC1  J - 0.75 ln(N) / T^0.3 df
                  BIC2  J - 0.75 ln(N) / ln(5) df            (the same as "bic2" in panel_tools.py)
                sequential test from r = 0 (page 6) with three levels: 5 percent (version 1),
                J1 = T / N and J2 = 10 / N (page 8).
                As in version 1, if every r is rejected the estimate is the next value, and the criteria are 0 there.
                The paper does not say what to do in that case. This is my choice.
  als_cu_paper  Appendix A (page 12) as printed: the matrix A is held fixed while beta and H are iterated.
                First A = I, then A = the second moment of the residuals at the first estimate.
                panel_tools.als_cu updates A with beta (the version of the lecture notes). With no regressor both are the same.
  als_gmm_v2    the two-step GMM of equation (7), page 5: the weighting matrix is the second moment of the moment
                functions, not demeaned (the paper says why), at a first consistent estimate. Here the first
                estimate is the continuous-updating one. Normalization: the last r periods (equation 3).
B. Bai (2009), Econometrica 77, 1229-1279, and Moon and Weidner (2017), Econometric Theory 33, 158-195.
  bai_ife_v2    two start values (OLS and beta = 0); the smaller sum of squared residuals wins
                (Moon and Weidner 2017, page 165: several local minima may exist).
                Bias correction of Bai's Section 7, equations (22) to (24): beta - B/N - C/T.
                Standard errors: White (D_3 of Bai's Theorem 4), the lecture notes' version, and clustered by period (mine).
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.stats import chi2

from panel_tools import _c, als_cu, cc2, fe_ols, within2  # unchanged pieces


def pick_v2(J, df, N, T):
    rs = sorted(J)
    cap = rs[-1] + 1
    seq = lambda level: next((r for r in rs if chi2.sf(J[r], df[r]) >= level), cap)

    def crit(pen):
        c = {r: J[r] - pen * df[r] for r in rs}
        c[cap] = 0.0
        return min(c, key=c.get)

    return {"seq05": seq(0.05), "seqJ1": seq(T / N), "seqJ2": seq(10.0 / N), "aic": crit(2.0), "bic": crit(np.log(N)),
            "bic1": crit(0.75 * np.log(N) / T ** 0.3), "bic2": crit(0.75 * np.log(N) / np.log(5))}


def als_count_v2(Y, X, S):
    """Continuous-updating J for each r (version of the lecture notes), then all estimates of pick_v2."""
    N, T = Y.shape
    q = S.shape[1]
    k = 0 if X is None else X.shape[2]
    J, df, beta = {}, {}, {}
    for r in range(min(T, q)):
        if (T - r) * (q - r) - k <= 0:
            break
        beta[r], J[r], df[r] = als_cu(Y, X, S, r)
    out = pick_v2(J, df, N, T)
    out.update({"J": J, "df": df, "beta": beta})
    return out


def als_cu_paper(Y, X, S, r, tol=1e-10, max_iter=1000):
    """Appendix A of the paper for a model with regressors. Returns beta, J, df."""
    N, T = Y.shape
    q, k = S.shape[1], X.shape[2]
    Yc, Sc, Xc = _c(Y), _c(S), X - X.mean(0)
    Wi = np.linalg.inv(Sc.T @ Sc / N)
    sy = Sc.T @ Yc / N
    sx = [Sc.T @ Xc[:, :, j] / N for j in range(k)]

    def run(A, beta):
        for _ in range(max_iter):
            O = sy - sum(b * s for b, s in zip(beta, sx))          # q x T, moments of w and u(beta)
            val, V = eigh(O.T @ Wi @ O, A)                          # generalized eigenvalues, smallest first
            H = V[:, :T - r]                                        # H'AH = I
            c = (sy @ H).T.ravel()
            D = np.column_stack([(s @ H).T.ravel() for s in sx])
            W = np.kron(np.eye(T - r), Wi)
            new = np.linalg.solve(D.T @ W @ D, D.T @ W @ c)
            done = np.max(np.abs(new - beta)) < tol
            beta = new
            if done:
                break
        O = sy - sum(b * s for b, s in zip(beta, sx))
        return beta, float(N * eigh(O.T @ Wi @ O, A, eigvals_only=True)[:T - r].sum())

    b1, _ = run(np.eye(T), fe_ols(Y, X))
    U = Yc - Xc @ b1
    b2, J = run(U.T @ U / N, b1)
    return b2, J, (T - r) * (q - r) - k


def als_gmm_v2(Y, X, S, r, order=None, centered=False):
    """Two-step GMM as in equation (7) of the paper. Returns beta (or None), J, df."""
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

    p0 = np.zeros(k + (T - r) * r)                       # first estimate: continuous updating
    if k:
        p0[:k] = als_cu(Y, X, S, r)[0]
    if r:
        U = Yc if k == 0 else Yc - Xc @ p0[:k]
        O = Sc.T @ U / N
        V = eigh(O.T @ np.linalg.inv(Sc.T @ Sc / N) @ O, U.T @ U / N)[1][:, :T - r]
        p0[k:] = (-(V[b] @ np.linalg.inv(V[a]))).T.ravel()
    G = mom(p0)
    if centered:
        G = G - G.mean(0)
    W = np.linalg.inv(G.T @ G / N)
    val, vec = np.linalg.eigh(W)
    C = vec * np.sqrt(np.clip(val, 0, None))
    p1 = p0 if len(p0) == 0 else least_squares(lambda p: C.T @ mom(p).mean(0), p0, method="lm").x
    g = mom(p1).mean(0)
    return (p1[:k] if k else None), float(N * g @ W @ g), (T - r) * (q - r) - k


def bai_ife_v2(Y, X, r, tol=1e-9, max_iter=5000):
    """Returns a dict: beta, beta_bc (bias corrected), se_white, se_slides, se_cluster, ssr of the two starts."""
    N, T = Y.shape
    k = X.shape[2]
    y = within2(Y)
    x = np.stack([within2(X[:, :, j]) for j in range(k)], axis=2)
    ols = np.linalg.lstsq(x.reshape(-1, k), y.ravel(), rcond=None)[0]

    def proj(beta):
        U, s, Vt = np.linalg.svd(y - x @ beta, full_matrices=False)
        L, F = U[:, :r], Vt[:r].T
        pl = lambda A: A - L @ (L.T @ A)
        yt = pl(y) - (pl(y) @ F) @ F.T
        xt = np.stack([pl(x[:, :, j]) - (pl(x[:, :, j]) @ F) @ F.T for j in range(k)], axis=2)
        return yt, xt, L, F, s

    def run(beta):
        for _ in range(max_iter):
            yt, xt, L, F, s = proj(beta)
            new = np.linalg.lstsq(xt.reshape(-1, k), yt.ravel(), rcond=None)[0]
            done = np.max(np.abs(new - beta)) < tol
            beta = new
            if done:
                break
        return beta, float((proj(beta)[4][r:] ** 2).sum())

    if r == 0:
        beta, ssr = ols, [float(((y - x @ ols) ** 2).sum())] * 2
        yt, xt = y, x
    else:
        fits = [run(ols), run(np.zeros(k))]
        ssr = [f[1] for f in fits]
        beta = fits[int(np.argmin(ssr))][0]
        yt, xt, L, F, s = proj(beta)
    e = yt - xt @ beta
    D0 = np.einsum("itj,itl->jl", xt, xt) / (N * T)
    bread = np.linalg.inv(D0 * N * T)
    adj = N * T / max((N - r - 1) * (T - r - 1) - k, 1)
    se = lambda meat: np.sqrt(np.diag(bread @ meat @ bread) * adj)
    gt = np.einsum("itj,it->tj", xt, e)
    out = {"beta": beta, "ssr": ssr, "se_white": se(np.einsum("itj,itl,it->jl", xt, xt, e ** 2)),
           "se_slides": se(np.einsum("itj,itl,i->jl", xt, xt, (e ** 2).mean(1))), "se_cluster": se(gt.T @ gt), "beta_bc": beta}
    if r > 0:
        Fb = np.sqrt(T) * F                                   # Bai's normalization F'F / T = I
        Lam = L * s[:r] / np.sqrt(T)                          # loadings, N x r
        LLi = np.linalg.inv(Lam.T @ Lam / N)
        mlx = np.stack([x[:, :, j] - L @ (L.T @ x[:, :, j]) for j in range(k)], axis=2)       # X_i - V_i
        mfx = np.stack([x[:, :, j] - (x[:, :, j] @ F) @ F.T for j in range(k)], axis=2)       # M_F X_i
        B = -np.linalg.solve(D0, np.einsum("itj,tr,rs,is,i->j", mlx, Fb, LLi, Lam, (e ** 2).mean(1)) / (N * T))
        Cc = -np.linalg.solve(D0, np.einsum("itj,t,tr,rs,is->j", mfx, (e ** 2).mean(0), Fb, LLi, Lam) / (N * T))
        out["beta_bc"] = beta - B / N - Cc / T
    return out
