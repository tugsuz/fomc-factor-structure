"""Shared tools for the Ahn and Zhang (2023) multilevel group factor model (9 Nov 2026).

Paper: Ahn and Zhang, "Model Selection for General Multi-Level Group Factor Models with Global,
Regional and Local Factors" (SSRN 4503954, April 2023 draft). The model has R known groups.
Global factors load on every group, a level-k regional factor on exactly k groups, a local factor
on one group. The tools here implement:

  projection_estimators  ProjectionMER and ProjectionET for the number of global factors
                         (Section 3.2, eqs. 3.5 to 3.10) and the global factor estimator (3.11).
  algorithm1             Algorithm 1 (Section 4.1): global count, concentrate the global factors
                         out, search the subsets of k groups for k = R-1, ..., 2 for regional
                         factors, concentrate them out, then count the local factors in each group
                         with the Ahn and Horenstein (2013) ER and GR estimators.
  variance_shares        share of each group's variance explained by the global, regional and
                         local components (sequential projections).

Same formulas as code/multilevel_projection_check.py (first run on daily CRSP stocks), with two
changes. (1) Speed: W = sum_i P(K_i) is T x T. Its nonzero eigenvalues equal those of the small
Gram matrix U'U, where U = [U_1, ..., U_R] stacks the orthonormal top-rmax principal component
bases of the groups. So W is never formed; a subset of groups uses the matching sub-block of the
same Gram matrix. (2) Regional step: the paper evaluates all subsets of a level at once and does
not say what to do when two accepted regions on the same level overlap in part. Under Assumption F
(regions nested or non-overlapping) we accept the candidates in order of strength (smallest gap
between k and the top eigenvalue) and skip a candidate that overlaps an accepted region in part.
Without Assumption F every candidate with a positive count is accepted.

Notation: r is a number of factors. Data blocks are T x M_i arrays (rows are days). Every block
is demeaned series by series before the principal components are taken, as in the paper.
"""
import itertools

import numpy as np

from factor_count import ahn_horenstein

KMAX = 8


def series_demean(Z):
    Z = np.asarray(Z, dtype=float)
    return Z - Z.mean(axis=0, keepdims=True)


def partial_overlap(S, Q):
    """True when the two index sets overlap but neither contains the other."""
    S, Q = set(S), set(Q)
    inter = S & Q
    return 0 < len(inter) < min(len(S), len(Q))


def paper_rmax(blocks, kmax=KMAX):
    """The paper's choice: max over groups of the ER count, plus one (at least 1)."""
    r = [ahn_horenstein(series_demean(b), kmax=kmax, demean=False)["k_er"] for b in blocks]
    return max(1, max(r) + 1), r


def level_fit(blocks, rmax):
    """Per-group principal component bases and residual sums of squares, plus the Gram matrix U'U."""
    Ys = [series_demean(b) for b in blocks]
    T = Ys[0].shape[0]
    Us, sse, syy, sizes = [], [], [], []
    for Y in Ys:
        r_eff = int(min(rmax, Y.shape[1] - 1, T - 1))
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        Ub = U[:, :r_eff]
        Us.append(Ub)
        sse.append(float(((Y - Ub @ (Ub.T @ Y)) ** 2).sum()))
        syy.append(float((Y ** 2).sum()))
        sizes.append(Y.shape[1])
    Ustack = np.hstack(Us)
    gram = Ustack.T @ Ustack
    offsets = np.cumsum([0] + [U.shape[1] for U in Us])
    return {"U": Us, "gram": gram, "offsets": offsets, "sse": np.array(sse), "syy": np.array(syy), "sizes": np.array(sizes), "T": T, "rmax": rmax}


def subset_estimators(fit, S=None):
    """ProjectionMER and ProjectionET for the groups in S (default: all groups) from a level_fit."""
    R_all = len(fit["U"])
    S = tuple(range(R_all)) if S is None else tuple(S)
    R = len(S)
    T = fit["T"]
    M = int(fit["sizes"][list(S)].min())
    cols = np.concatenate([np.arange(fit["offsets"][i], fit["offsets"][i + 1]) for i in S])
    gram = fit["gram"][np.ix_(cols, cols)]
    ev, V = np.linalg.eigh(gram)
    order = np.argsort(ev)[::-1]
    rho = np.clip(ev[order], 0.0, None)
    V = V[:, order]
    noise = float(fit["sse"][list(S)].sum() / fit["syy"][list(S)].sum())
    delta = min(np.sqrt(M), np.sqrt(T))
    C = (np.log(R) / 2.0) * np.exp(noise)
    ell = C / delta
    kappa = C * np.sqrt(M + T) / np.sqrt(M * T) * np.log(delta)
    rstar = int(((rho >= R - 1) & (rho <= R + 1e-9)).sum())
    gaps = np.maximum(R - np.concatenate([[R - ell], rho]), 1e-12)      # gaps[0] is the mock gap ell
    crit = [float(gaps[r + 1] / gaps[r]) for r in range(0, rstar + 1)]
    r_mer = int(np.argmax(crit))
    r_et = int((rho > R - kappa).sum())
    G = None
    if r_mer > 0:
        Ustack = np.hstack([fit["U"][i] for i in S])
        g = Ustack @ V[:, :r_mer] / np.sqrt(rho[:r_mer])
        G = np.sqrt(T) * g
    return {"S": S, "R": R, "T": T, "M": M, "rmax": fit["rmax"], "rho": rho, "rstar": rstar, "ell": ell, "kappa": kappa, "crit": crit,
            "r_mer": r_mer, "r_et": r_et, "G": G, "noise_share": noise, "gap1": float(R - rho[0]) if len(rho) else np.nan}


def projection_estimators(blocks, rmax=None):
    """ProjectionMER and ProjectionET for a list of T x M_i blocks. rmax None = the paper's rule."""
    if rmax is None:
        rmax, _ = paper_rmax(blocks)
    fit = level_fit(blocks, rmax)
    return subset_estimators(fit)


def concentrate(blocks, G):
    """M(G) Y_i for every block (series-demeaned first); unchanged when G is None."""
    Ys = [series_demean(b) for b in blocks]
    if G is None:
        return Ys
    Q, _ = np.linalg.qr(G)
    return [Y - Q @ (Q.T @ Y) for Y in Ys]


def local_counts(blocks, kmax=KMAX):
    out = []
    for Y in blocks:
        o = ahn_horenstein(Y, kmax=kmax, demean=False)
        o2 = ahn_horenstein(Y, kmax=kmax, demean=True)
        mu = o["eigenvalues"]
        out.append({"r_er": o["k_er"], "r_gr": o["k_gr"], "er1": float(o["er"][1]), "er2": float(o["er"][2]), "share1": float(mu[0] / mu.sum()),
                    "r_er_dd": o2["k_er"], "r_gr_dd": o2["k_gr"], "er1_dd": float(o2["er"][1])})
    return out


def algorithm1(blocks, rmax=None, assumption_F=True, regional=True, kmax=KMAX, rmax_regional=None):
    """Algorithm 1 of Ahn and Zhang (2023). Returns the global result, the regions, the local counts and a log of every candidate subset.

    rmax: components per group for the global step (None = the paper's rule). rmax_regional: the same for the
    regional steps (None = same as rmax). With regional=False only the global and local steps run.
    """
    R = len(blocks)
    if rmax is None:
        rmax, _ = paper_rmax(blocks, kmax)
    rmax_regional = rmax if rmax_regional is None else rmax_regional
    fit0 = level_fit(blocks, rmax)
    glob = subset_estimators(fit0)
    cur = concentrate(blocks, glob["G"])
    regions, log = [], []
    if regional:
        for k in range(R - 1, 1, -1):
            fit = level_fit(cur, rmax_regional)
            found = []
            for S in itertools.combinations(range(R), k):
                if assumption_F and any(partial_overlap(S, q["groups"]) for q in regions):
                    continue
                o = subset_estimators(fit, S)
                rec = {"level": k, "groups": S, "rho1": float(o["rho"][0]), "rho2": float(o["rho"][1]) if len(o["rho"]) > 1 else np.nan,
                       "gap1": o["gap1"], "ell": o["ell"], "kappa": o["kappa"], "rstar": o["rstar"], "r_mer": o["r_mer"], "r_et": o["r_et"], "accepted": False}
                log.append(rec)
                if o["r_mer"] > 0:
                    found.append((S, o, rec))
            found.sort(key=lambda t: t[1]["gap1"])
            accepted = []
            for S, o, rec in found:
                if assumption_F and any(partial_overlap(S, A) for A, _, _ in accepted):
                    continue
                accepted.append((S, o, rec))
            for S, o, rec in accepted:
                rec["accepted"] = True
                regions.append({"level": k, "groups": S, "r": o["r_mer"], "r_et": o["r_et"], "rho": o["rho"][:3], "G": o["G"]})
                Q, _ = np.linalg.qr(o["G"])
                for i in S:
                    cur[i] = cur[i] - Q @ (Q.T @ cur[i])
    local = local_counts(cur, kmax)
    return {"global": glob, "regions": regions, "local": local, "log": log, "concentrated": cur, "rmax": rmax}


def variance_shares(blocks, res):
    """Share of each block's (series-demeaned) sum of squares explained by the global factors, by the regional
    factors that include the block, and by the block's own local factors (top r_er components of the fully
    concentrated block). The remainder is idiosyncratic."""
    out = []
    G = res["global"]["G"]
    QG = np.linalg.qr(G)[0] if G is not None else None
    for i, b in enumerate(blocks):
        Y = series_demean(b)
        tot = float((Y ** 2).sum())
        cur = Y
        g = 0.0
        if QG is not None:
            PY = QG @ (QG.T @ cur)
            g = float((PY ** 2).sum()) / tot
            cur = cur - PY
        h = 0.0
        for reg in res["regions"]:
            if i in reg["groups"]:
                Q = np.linalg.qr(reg["G"])[0]
                PY = Q @ (Q.T @ cur)
                h += float((PY ** 2).sum()) / tot
                cur = cur - PY
        r_loc = res["local"][i]["r_er"]
        l = 0.0
        if r_loc > 0:
            s = np.linalg.svd(cur, compute_uv=False)
            l = float((s[:r_loc] ** 2).sum()) / tot
        out.append({"global": g, "regional": h, "local": l, "idiosyncratic": 1.0 - g - h - l})
    return out


def describe(res, names=None, digits=3):
    """Plain-text lines for a result of algorithm1."""
    g = res["global"]
    R = g["R"]
    names = names or [f"g{i}" for i in range(R)]
    lines = [f"global: r_MER = {g['r_mer']}, r_ET = {g['r_et']}, rmax = {g['rmax']}, eigenvalues of W: "
             + " ".join(f"{x:.{digits}f}" for x in g["rho"][:min(8, len(g['rho']))]) + f" (R = {R}), r* = {g['rstar']}, ell = {g['ell']:.3f}, kappa = {g['kappa']:.3f}, noise share {g['noise_share']:.3f}"]
    if res["regions"]:
        for reg in res["regions"]:
            lines.append(f"region level {reg['level']}: {{{', '.join(names[i] for i in reg['groups'])}}}: {reg['r']} factor(s) (MER; ET {reg['r_et']}), eigenvalues "
                         + " ".join(f"{x:.{digits}f}" for x in reg["rho"]))
    else:
        lines.append("regions: none")
    lines.append("local counts (AH on the concentrated groups, series demeaned; dd = double demeaned): "
                 + "; ".join(f"{names[i]}: r_ER {l['r_er']} r_GR {l['r_gr']} ER(1) {l['er1']:.2f} [dd r_ER {l['r_er_dd']}]" for i, l in enumerate(res["local"])))
    return lines
