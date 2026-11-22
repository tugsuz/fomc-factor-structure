"""Checks for the review of 19 Nov 2026: the twelve open points, recomputed where a number can settle them.

Notation: m = number of all factors (Ahn and Perez 2010), r = number of factors or effects, L = the number used
in one test. k is the number of regressors in the panel lecture notes and is not used here. k is never a number of factors.

Parts (python3 review_checks.py <letter>), output results/review_2026-11-19/review_checks_<letter>.txt
  A  Point 2. Can the criteria over-count, and what does an estimate at the cap mean? A simulation with the true
     number below the cap (N = 8 and 12; heavy event sizes, events scaled), and one with errors that are correlated
     between neighbouring series.
  B  Point 2. The rate counts again with larger panels, and with the same series for all three event types.
  C  Point 3. The J statistic with the plain (not demeaned) and the demeaned weighting matrix.
  D  Point 4. How much the split shares move with the seed, for 100 and 500 splits, and near-full enumeration.
  E  Point 1. Scaling each event by the size of the instrument group only, against scaling by the whole row.
  F  Points 5 and 6. AH (2013) on the merged panel without near duplicates; single and double demeaning.
  G  Point 8. Matched-block permutation P values for the gaps in m and r between statement and control days (ETFs).
  H  Point 12. The count in panel_perm.stats with and without the largest value.
  I  Point 2 again, monthly data: the count on 9, 12 and 16 size-BM portfolios.
  J  The two-market count (futures as responses, Treasury yields as candidates): a simulation at our sizes.
     Run as: python3 review_checks.py J <errors>, with errors = independent, neighbours, own or cross.
  K  The two-market count: statements against minutes. A permutation P value for the gap, and the canonical
     correlations between the two markets by event type.
  L  Robustness of K: the same sample period for both event types, press conferences, and the two markets
     with their roles exchanged.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

import ap_tools_v2 as ap
import paths

OUT = os.path.join(paths.RESULTS, "review_2026-11-19")
CRIT = ["seq", "bic1", "bic2", "bic3"]
S8 = ["MP1", "MP2", "ED2", "ED3", "ED4", "UST2Y", "UST5Y", "UST10Y"]
C10 = ["MP2", "FF4", "ED2", "ED4", "ED6", "ED8", "UST2Y", "UST5Y", "UST10Y", "UST30Y"]
R12 = ["MP1", "MP2", "FF4", "ED2", "ED3", "ED4", "ED6", "ED8", "UST2Y", "UST5Y", "UST10Y", "UST30Y"]
FUT6 = ["MP2", "FF4", "ED2", "ED4", "ED6", "ED8"]
UST4 = ["UST2Y", "UST5Y", "UST10Y", "UST30Y"]
UST6 = ["UST3M", "UST6M", "UST2Y", "UST5Y", "UST10Y", "UST30Y"]


def std(X):
    return (X - X.mean(0)) / X.std(0, ddof=1)


def scaled(X):
    Z = std(X)
    return Z / np.sqrt((Z ** 2).mean(1, keepdims=True))


def usmpd(sheet):
    u = pd.read_excel(os.path.join(paths.FED, "USMPD.xlsx"), sheet)
    if "Unscheduled" in u:
        u = u[u["Unscheduled"] == 0]
    return u


def panel(u, cols):
    return u[cols].astype(float).dropna().values


def shares(v):
    v = np.asarray(v, float)
    vals, cnt = np.unique(v[~np.isnan(v)], return_counts=True)
    order = np.argsort(-cnt)
    return ", ".join(f"{int(vals[i])}: {cnt[i] / cnt.sum():.2f}" for i in order[:4])


def mode(v):
    vals, cnt = np.unique(v[~np.isnan(v)], return_counts=True)
    return vals[np.argmax(cnt)]


def count_perms(X, perms, centered=False, inst_scale=False):
    """As ap.count_m, for a given list of splits. inst_scale: X is standardized by series only, and each event is
    divided by the root mean square of the instrument group of the split."""
    T, N = X.shape
    Q = min(N // 2, 10)
    P = N - Q
    out = {c: [] for c in CRIT}
    Js = []
    for perm in perms:
        g, z = X[:, perm[:P]], X[:, perm[P:]]
        if inst_scale:
            s = np.sqrt((z ** 2).mean(1, keepdims=True))
            g, z = g / s, z / s
        J = {l: ap.gmm_j(g[:, :P - l], g[:, P - l:] if l else None, z, 0, centered)[0] for l in range(Q)}
        Js.append([J[l] for l in range(Q)])
        sel = ap.select(J, P, Q, T)
        for c in CRIT:
            out[c].append(sel[c])
    return {c: np.array(v, float) for c, v in out.items()}, np.array(Js)


def rand_perms(N, n, rng):
    return [rng.permutation(N) for _ in range(n)]


def line(res, keys=CRIT):
    return "   ".join(f"{c} {{{shares(res[c])}}}" for c in keys)


# ---------------------------------------------------------------- A
def part_a():
    rng = np.random.default_rng(2026111901)
    u = usmpd("Statements")
    lines = ["Simulation. The model is fitted to the scaled statement panel: loadings from its first m0 principal components,",
             "independent errors with the fitted variances, event sizes drawn from the data, then each event scaled.",
             "For each panel: the most frequent estimate over 30 random splits. 100 panels in each cell.",
             "Columns: mean estimate (share of panels with the true number m0). The cap of the criteria is N // 2 - 1.",
             "'neighbours': the errors of neighbouring series are correlated (0.6 between neighbours, in the order of the panel).", ""]
    for name, cols in [("S8", S8), ("R12", R12)]:
        X = panel(u, cols)
        N = len(cols)
        s = np.sqrt((std(X) ** 2).mean(1))
        C = np.cov(scaled(X).T)
        val, vec = np.linalg.eigh(C)
        val, vec = val[::-1], vec[:, ::-1]
        lines.append(f"{name}: N = {N}, cap {N // 2 - 1}. Shares of the first five components in the data: {np.round(val[:5] / val.sum(), 3).tolist()}")
        for m0 in ([2, 3] if N == 8 else [2, 3, 4]):
            Lam = vec[:, :m0] * np.sqrt(val[:m0])
            psi = np.clip(np.diag(C - Lam @ Lam.T), 1e-4, None)
            for errors in ["independent", "neighbours"]:
                if errors == "neighbours" and N == 8 and m0 == 3:
                    continue
                R = 0.6 ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
                A = np.linalg.cholesky(R if errors == "neighbours" else np.eye(N))
                for T in [166, 260]:
                    est = {c: [] for c in CRIT}
                    for _ in range(100):
                        e = (rng.standard_normal((T, N)) @ A.T) * np.sqrt(psi)
                        x = rng.choice(s, T)[:, None] * (rng.standard_normal((T, m0)) @ Lam.T + e)
                        res = ap.count_m(scaled(x), 30, 0, rng)
                        for c in CRIT:
                            est[c].append(mode(res[c]))
                    lines.append(f"   m0 {m0}  errors {errors:11s} T {T}:  " +
                                 "   ".join(f"{c} {np.mean(est[c]):.2f} ({np.mean(np.array(est[c]) == m0):.2f})" for c in CRIT))
    return lines


# ---------------------------------------------------------------- B
def part_b():
    rng = np.random.default_rng(2026111902)
    lines = ["Rate counts on USMPD, scheduled events, each event scaled. 500 random splits. Tables are {estimate: share of splits}.",
             "S8 = MP1 MP2 ED2 ED3 ED4 UST2Y UST5Y UST10Y (cap 3). C10 = MP2 FF4 ED2 ED4 ED6 ED8 UST2Y UST5Y UST10Y UST30Y (cap 4;",
             "the same ten series exist for all three event types). R12 = S8 plus FF4 ED6 ED8 UST30Y (cap 5).", ""]
    for sheet in ["Statements", "Minutes", "Press Conferences"]:
        u = usmpd(sheet)
        for name, cols in [("S8", S8), ("C10", C10), ("R12", R12)]:
            if sheet != "Statements" and name != "C10":
                continue
            X = panel(u, cols)
            res, _ = count_perms(scaled(X), rand_perms(len(cols), 500, rng))
            lines.append(f"{sheet:18s} {name:4s} T {len(X)} N {len(cols)} cap {len(cols) // 2 - 1}:  {line(res)}")
    lines += ["", "Two markets (the count with candidate factors): futures as responses, Treasury yields as candidates.",
              "FUT6 = MP2 FF4 ED2 ED4 ED6 ED8. UST4 = 2, 5, 10, 30 years (cap 3). UST6 adds 3 and 6 months (cap 5). 300 orders."]
    for sheet in ["Statements", "Minutes", "Press Conferences"]:
        u = usmpd(sheet)
        for zname, zc in [("UST4", UST4), ("UST6", UST6)]:
            d = u[FUT6 + zc].astype(float).dropna()
            W = scaled(d.values)
            res = ap.count_r(W[:, :6], W[:, 6:], 300, 0, rng)
            lines.append(f"{sheet:18s} FUT6 on {zname} T {len(d)}:  {line(res)}")
    return lines


# ---------------------------------------------------------------- C
def part_c():
    rng = np.random.default_rng(2026111903)
    u = usmpd("Statements")
    lines = ["J statistics on the statement panels: median over 200 splits, for L = 0, 1, 2, ... The plain weighting matrix is the",
             "second moment of the moment functions. Then J = T a / (1 + a), where T a is the J of the demeaned matrix. So J < T.", ""]
    for name, cols in [("S8", S8), ("R12", R12)]:
        X = panel(u, cols)
        perms = rand_perms(len(cols), 200, rng)
        for vname, V in [("raw (standardized series)", std(X)), ("events scaled", scaled(X))]:
            for cen in [False, True]:
                res, Js = count_perms(V, perms, centered=cen)
                lines.append(f"{name} {vname:26s} T {len(X)} {'demeaned' if cen else 'plain   '}: median J {np.round(np.nanmedian(Js, 0), 1).tolist()}   " +
                             line(res, ["bic1", "bic2", "seq"]))
    return lines


# ---------------------------------------------------------------- D
def part_d():
    u = usmpd("Statements")
    W = scaled(panel(u, S8))
    lines = ["S8, scheduled statements, events scaled. Share of splits with the BIC2 estimate 3 and with the sequential estimate 3.", ""]
    for n in [100, 500]:
        a, b = [], []
        for seed in range(20):
            res, _ = count_perms(W, rand_perms(8, n, np.random.default_rng(seed)))
            a.append(np.mean(res["bic2"] == 3))
            b.append(np.mean(res["seq"] == 3))
        lines.append(f"{n} random splits, 20 seeds: BIC2 share of 3 from {min(a):.2f} to {max(a):.2f} (sd {np.std(a):.3f});  seq share of 3 from {min(b):.2f} to {max(b):.2f} (sd {np.std(b):.3f})")
    rng = np.random.default_rng(0)
    perms = []
    for g in itertools.combinations(range(8), 4):
        rest = [i for i in range(8) if i not in g]
        for _ in range(6):
            perms.append(np.array(list(rng.permutation(g)) + list(rng.permutation(rest))))
    res, _ = count_perms(W, perms)
    lines.append(f"All 70 ways to choose group one, 6 random orders inside each (420 splits): {line(res)}")
    return lines


# ---------------------------------------------------------------- E
def part_e():
    rng = np.random.default_rng(2026111905)
    u = usmpd("Statements")
    lines = ["Scaling by the whole row against scaling by the instrument group only (the size then does not depend on the",
             "series whose errors enter the moment conditions). 500 splits.", ""]
    for name, cols in [("S8", S8), ("C10", C10), ("R12", R12)]:
        X = panel(u, cols)
        perms = rand_perms(len(cols), 500, rng)
        r1, _ = count_perms(scaled(X), perms)
        r2, _ = count_perms(std(X), perms, inst_scale=True)
        lines += [f"{name} whole row:        {line(r1)}", f"{name} instrument group: {line(r2)}"]
    return lines


# ---------------------------------------------------------------- F
def part_f():
    import rates_big_panel as rb
    from factor_count_v2 import ahn_horenstein, double_demean
    from rates_dd_adp import FLIP
    E, u, st, A, B = rb.load()
    ucols = [c for g in rb.U_GROUPS.values() for c in g]
    ecols = [c for g in rb.E_GROUPS.values() for c in g]
    days = [d for d in st if d >= "2008-01-01" and d in u.index and d in E.index and u.loc[d, ucols].notna().all() and E.loc[d, ecols].notna().all()]
    D = pd.concat([u.loc[days, ucols].astype(float), E.loc[days, ecols].astype(float)], axis=1)
    for c in D.columns:
        if c in FLIP:
            D[c] = -D[c]
    sets = {"all 43": list(D.columns),
            "no near duplicates (18)": ["MP1", "ED2", "ED4", "ED8", "UST6M", "UST2Y", "UST5Y", "UST10Y", "UST30Y", "TIPS5Y", "TIPS10Y", "DXY", "USDJPY",
                                        "E_LQD", "E_HYG", "E_MBB", "E_MUB", "E_AGG"],
            "USMPD 27": ucols, "USMPD without near duplicates (13)": ["MP1", "ED2", "ED4", "ED8", "UST6M", "UST2Y", "UST5Y", "UST10Y", "UST30Y",
                                                                       "TIPS5Y", "TIPS10Y", "DXY", "USDJPY"]}
    C = D.corr().abs().values
    np.fill_diagonal(C, 0)
    lines = [f"Statement days 2008 to 2025 in both sources: {len(days)}. Signs aligned as in rates_dd_adp.py. Each series standardized, each event scaled.",
             f"Pairs of series with an absolute correlation above 0.95 in the panel of 43: {int((C > 0.95).sum() / 2)}; above 0.90: {int((C > 0.90).sum() / 2)}.",
             "sd = series demeaned only. dd = double demeaned. ER and GR counts, ER(1), ER(2), and the first three eigenvalue shares.", ""]
    for name, cols in sets.items():
        Z = rb.scale_rows(rb.std(D[cols].values))
        for tag, V in [("sd", Z), ("dd", double_demean(Z))]:
            o = ahn_horenstein(V, kmax=min(8, len(cols) // 2), demean=False)
            sh = o["eigenvalues"][:3] / o["eigenvalues"].sum()
            lines.append(f"{name:36s} {tag}: ER {o['r_er']} GR {o['r_gr']}  ER(1) {o['er'][1]:.2f} ER(2) {o['er'][2]:.2f}  shares {' '.join(f'{x:.3f}' for x in sh)}")
    return lines


# ---------------------------------------------------------------- G
def part_g(n_draw=299):
    import rates_big_panel as rb
    rng = np.random.default_rng(2026111907)
    E, u, st, A, B = rb.load()
    ETF8 = ["E_SHY", "E_IEI", "E_IEF", "E_TLT", "E_TIP", "E_AGG", "E_LQD", "E_MBB"]
    CRED = ["E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB", "E_FXE", "E_UUP"]
    TRE = ["E_SHY", "E_IEF", "E_TLT", "E_TIP"]
    lines = ["Matched-block permutation for the ETF panels (TAQ window returns, 2008 to 2025). A block is a scheduled statement day",
             "with its own control days. A draw takes one day from each block. Each event is scaled. The same 30 splits are used for",
             f"every draw. P = (1 + number of draws at least as large) / (1 + {n_draw}).",
             "Statistics: mean over the splits of the BIC2 estimate and of the sequential estimate; share of splits at the cap;",
             "the share of the first eigenvalue.", ""]
    for name, cols, kind in [("m, 8 bond ETFs (cap 3)", ETF8, "m"), ("r, 9 credit and FX ETFs on 4 Treasury and TIPS ETFs (cap 3)", CRED + TRE, "r")]:
        ok = lambda d: d in E.index and E.loc[d, cols].notna().all()
        blocks = [[d] + [x for x in (A.get(d), B.get(d)) if x is not None and ok(x)] for d in st if d >= "2008-01-01" and ok(d)]
        blocks = [b for b in blocks if len(b) > 1]
        V = E[cols].astype(float)
        perms = rand_perms(len(cols) if kind == "m" else len(CRED), 30, np.random.default_rng(1))

        def stat(days):
            W = scaled(V.loc[days].values)
            if kind == "m":
                res, _ = count_perms(W, perms)
                cap = len(cols) // 2 - 1
            else:
                T = len(W)
                res = {c: [] for c in CRIT}
                for p in perms:
                    y = W[:, :len(CRED)][:, p]
                    J = {l: ap.gmm_j(y[:, :len(CRED) - l], y[:, len(CRED) - l:] if l else None, W[:, len(CRED):], 0, False)[0] for l in range(len(TRE))}
                    sel = ap.select(J, len(CRED), len(TRE), T)
                    for c in CRIT:
                        res[c].append(sel[c])
                res = {c: np.array(v, float) for c, v in res.items()}
                cap = len(TRE) - 1
            ev = np.linalg.eigvalsh(np.cov(W.T))[::-1]
            return np.array([res["bic2"].mean(), res["seq"].mean(), np.mean(res["bic2"] >= cap), ev[0] / ev.sum()])
        obs = stat([b[0] for b in blocks])
        draws = np.array([stat([b[rng.integers(len(b))] for b in blocks]) for _ in range(n_draw)])
        p = (1 + (draws >= obs - 1e-12).sum(0)) / (1 + n_draw)
        lines.append(f"{name}: {len(blocks)} blocks")
        for j, s in enumerate(["mean BIC2 estimate", "mean sequential estimate", "share of splits at the cap (BIC2)", "first eigenvalue share"]):
            lines.append(f"   {s:34s} statement {obs[j]:.3f}   draws {draws[:, j].mean():.3f}   P {p[j]:.3f}")
    return lines


# ---------------------------------------------------------------- H
def part_h(n_draw=999):
    import panel_perm as pp
    import panel_tools as pt
    rng = np.random.default_rng(2026111908)
    d = pt.stock_windows()
    st, A, B = pt.day_sets(d)
    tr = pd.read_csv(os.path.join(pp.OUT, "traits_v2.csv"))
    for s in pp.SECT:
        tr[s] = (tr["sector"] == s).astype(float)

    def count(Y, Qs, q, largest):
        N, T = Y.shape
        c2 = np.linalg.svd(np.linalg.qr(Y - Y.mean(0))[0].T @ Qs, compute_uv=False) ** 2
        pen = 0.75 * np.log(N) / np.log(5)
        tail = np.cumsum(c2[::-1])[::-1]
        vals = [N * tail[r] - pen * (T - r) * (q - r) for r in range(min(T, q))] + ([0.0] if largest else [])
        return int(np.argmin(vals))
    lines = ["Count by BIC2 with risk traits (q = 5, T = 8). 'with' allows the largest value 5, where the criterion is 0",
             "(as in panel_perm.stats). 'without' takes the minimum over 0 to 4, as the paper does (it sets a maximum below the largest value).", ""]
    res = {}
    for y in range(2005, 2026):
        R, blocks, t = pp.year_blocks(d, st, A, B, y, tr)
        S = t[pp.RISK].values
        Qs = np.linalg.qr(S - S.mean(0))[0]
        f = lambda cols: [count(R[:, cols], Qs, 5, True), count(R[:, cols], Qs, 5, False)]
        res[y] = (f([b[0] for b in blocks]), np.array([f([b[rng.integers(len(b))] for b in blocks]) for _ in range(n_draw)]))
    for a, b in pp.PERIODS:
        ys = [y for y in res if a <= y <= b]
        o = np.mean([res[y][0] for y in ys], axis=0)
        dr = np.mean([res[y][1] for y in ys], axis=0)
        p = (1 + (dr >= o - 1e-12).sum(0)) / (1 + n_draw)
        lines.append(f"{a}-{b}: with: statement {o[0]:.3f} draws {dr[:, 0].mean():.3f} P {p[0]:.3f}   without: statement {o[1]:.3f} draws {dr[:, 1].mean():.3f} P {p[1]:.3f}")
    n5 = sum(res[y][0][0] == 5 for y in res)
    lines.append(f"Years with the count 5 on statement days: {n5} of {len(res)}.")
    return lines


# ---------------------------------------------------------------- I
def part_i():
    from monthly_counts import load_all
    rng = np.random.default_rng(2026111909)
    _, mly, _, fm, _ = load_all()
    B = mly["BM25"]
    pick = lambda sizes, bms: [5 * s + b for s in sizes for b in bms]
    sets = {"9 (cap 3)": pick([0, 2, 4], [0, 2, 4]), "12 (cap 5)": pick([0, 2, 4], [0, 1, 3, 4]), "16 (cap 7)": pick([0, 1, 3, 4], [0, 1, 3, 4])}
    lines = ["Monthly Ahn and Perez counts on subsets of the 25 size-BM portfolios (raw monthly returns, as in french_als_monthly.py).",
             "300 random splits. If the estimate rises with the cap, the 3 of the 9-portfolio panel is the cap and not a count.",
             f"Factor columns in the monthly factor file: {list(fm.columns)}", ""]
    for a, b in [("1964-01", "2025-12"), ("1994-01", "2025-12")]:
        for name, idx in sets.items():
            Y = B.loc[a:b].iloc[:, idx].dropna()
            res, _ = count_perms(Y.values, rand_perms(len(idx), 300, rng))
            lines.append(f"{a[:4]}-{b[:4]} size-BM {name:10s} T {len(Y)}:  {line(res)}")
        Y = B.loc[a:b].iloc[:, sets["16 (cap 7)"]].dropna()
        for cols in [["Mkt-RF", "SMB", "HML", "RMW"], [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "MOM", "UMD"] if c in fm.columns][:6]]:
            Z = fm.reindex(Y.index)[cols]
            ok = Z.notna().all(axis=1).values
            res = ap.count_r(Y.values[ok], Z.values[ok], 300, 0, rng)
            lines.append(f"{a[:4]}-{b[:4]} 16 portfolios on {len(cols)} candidates {cols} (cap {len(cols) - 1}) T {int(ok.sum())}:  {line(res)}")
    return lines


# ---------------------------------------------------------------- J
def two_market_errors(kind, n):
    """Correlation matrix of the errors of two markets with n series each (order: market one, then market two)."""
    I, Z = np.eye(n), np.zeros((n, n))
    if kind == "independent":
        return np.eye(2 * n)
    if kind == "neighbours":      # 0.6 between neighbours inside a market, nothing across markets
        R = 0.6 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        return np.block([[R, Z], [Z, R]])
    if kind == "own":             # one factor of its own in each market: errors correlated 0.5 inside a market
        R = 0.5 * np.ones((n, n)) + 0.5 * I
        return np.block([[R, Z], [Z, R]])
    if kind == "cross":           # series j of market one and series j of market two share a part of the error (0.6)
        return np.block([[I, 0.6 * I], [0.6 * I, I]])
    raise ValueError(kind)


def part_j(kind):
    rng = np.random.default_rng(2026111910 + ["independent", "neighbours", "own", "cross"].index(kind))
    u = usmpd("Statements")
    d = u[FUT6 + UST6].astype(float).dropna()
    X = d.values
    s = np.sqrt((std(X) ** 2).mean(1))
    C = np.cov(scaled(X).T)
    val, vec = np.linalg.eigh(C)
    val, vec = val[::-1], vec[:, ::-1]
    lines = ["Simulation of the two-market count. Responses: 6 futures. Candidates: 6 Treasury yields. The model is fitted to the",
             "scaled statement panel (FUT6 and UST6): loadings of the shared factors from the first r0 principal components of the",
             "12 series, errors with the fitted variances, event sizes drawn from the data, then each event scaled.",
             "For each panel: the most frequent estimate over 30 random orders. 100 panels in each cell.",
             "Columns: mean estimate (share of panels with the true number r0). The criteria can reach 5 here.",
             f"Errors: {kind}. independent = no correlation. neighbours = 0.6 between neighbours inside a market.",
             "own = a factor of its own in each market (0.5 inside a market). cross = series j of the two markets share",
             "a part of the error (0.6); this breaks the assumption that the errors of the two groups are uncorrelated.",
             f"Shares of the first five components in the data: {np.round(val[:5] / val.sum(), 3).tolist()}", ""]
    A = np.linalg.cholesky(two_market_errors(kind, 6))
    for r0 in [1, 2, 3]:
        Lam = vec[:, :r0] * np.sqrt(val[:r0])
        psi = np.clip(np.diag(C - Lam @ Lam.T), 1e-4, None)
        for T in [95, 199, 243]:
            est = {c: [] for c in CRIT}
            for _ in range(100):
                e = (rng.standard_normal((T, 12)) @ A.T) * np.sqrt(psi)
                x = rng.choice(s, T)[:, None] * (rng.standard_normal((T, r0)) @ Lam.T + e)
                W = scaled(x)
                res = ap.count_r(W[:, :6], W[:, 6:], 30, 0, rng)
                for c in CRIT:
                    est[c].append(mode(res[c]))
            lines.append(f"   r0 {r0}  T {T:3d}:  " +
                         "   ".join(f"{c} {np.mean(est[c]):.2f} ({np.mean(np.array(est[c]) == r0):.2f})" for c in CRIT))
    return lines


# ---------------------------------------------------------------- K
def part_k(n_draw=999):
    rng = np.random.default_rng(2026111911)
    W = {}
    for sheet in ["Statements", "Minutes", "Press Conferences"]:
        d = usmpd(sheet)[FUT6 + UST6].astype(float).dropna()
        W[sheet] = scaled(d.values)
    orders = [np.random.default_rng(100 + i).permutation(6) for i in range(30)]

    def counts(V):
        T = len(V)
        res = {c: [] for c in CRIT}
        for p in orders:
            y = V[:, :6][:, p]
            J = {l: ap.gmm_j(y[:, :6 - l], y[:, 6 - l:] if l else None, V[:, 6:], 0, False)[0] for l in range(6)}
            sel = ap.select(J, 6, 6, T)
            for c in CRIT:
                res[c].append(sel[c])
        return np.array([np.mean(res[c]) for c in CRIT])

    def cancor(V):
        return ap.cca(V[:, :6], V[:, 6:])

    lines = ["Two markets: 6 futures (FUT6) and 6 Treasury yields (UST6), 30-minute changes from USMPD, scheduled events, each event",
             "scaled inside its own event type. r = the number of factors in the futures that are correlated with the Treasury yields.",
             "", "1. Canonical correlations between the two markets, by event type (largest first)."]
    for sheet, V in W.items():
        lines.append(f"   {sheet:18s} T {len(V):3d}:  " + "  ".join(f"{x:.3f}" for x in cancor(V)))
    lines += ["", "2. Statements against minutes. Mean estimate over the same 30 orders, for each criterion.",
              f"   A draw mixes the scaled events of the two types at random and keeps the two group sizes. {n_draw} draws.",
              "   P = (1 + number of draws with a gap at least as large) / (1 + draws). One-sided: statements have more.",
              "   The null is that a scaled statement event and a scaled minutes event come from the same distribution."]
    S, M = W["Statements"], W["Minutes"]
    obs_s, obs_m = counts(S), counts(M)
    cc_s, cc_m = cancor(S), cancor(M)
    pool = np.vstack([S, M])
    gap, ccgap = [], []
    for _ in range(n_draw):
        idx = rng.permutation(len(pool))
        a, b = pool[idx[:len(S)]], pool[idx[len(S):]]
        gap.append(counts(a) - counts(b))
        ccgap.append(cancor(a) - cancor(b))
    gap, ccgap = np.array(gap), np.array(ccgap)
    for j, c in enumerate(CRIT):
        p = (1 + (gap[:, j] >= obs_s[j] - obs_m[j] - 1e-12).sum()) / (1 + n_draw)
        lines.append(f"   {c:5s} statements {obs_s[j]:.3f}   minutes {obs_m[j]:.3f}   gap {obs_s[j] - obs_m[j]:+.3f}   "
                     f"draws: mean gap {gap[:, j].mean():+.3f}, 95th percentile {np.quantile(gap[:, j], 0.95):+.3f}   P {p:.3f}")
    lines += ["", "3. The same draws for each canonical correlation (statements minus minutes)."]
    for j in range(6):
        p = (1 + (ccgap[:, j] >= cc_s[j] - cc_m[j] - 1e-12).sum()) / (1 + n_draw)
        lines.append(f"   canonical correlation {j + 1}: statements {cc_s[j]:.3f}   minutes {cc_m[j]:.3f}   gap {cc_s[j] - cc_m[j]:+.3f}   "
                     f"draws: 95th percentile of the gap {np.quantile(ccgap[:, j], 0.95):+.3f}   P {p:.3f}")
    return lines


# ---------------------------------------------------------------- L
def part_l(n_draw=499):
    rng = np.random.default_rng(2026111912)
    orders = [np.random.default_rng(100 + i).permutation(6) for i in range(30)]

    def get(sheet, start, swap=False):
        u = usmpd(sheet)
        d = u[u["Date"] >= start][FUT6 + UST6].astype(float).dropna()
        V = scaled(d.values)
        return V[:, list(range(6, 12)) + list(range(6))] if swap else V

    def bic2(V):
        T, est = len(V), []
        for p in orders:
            y = V[:, :6][:, p]
            J = {l: ap.gmm_j(y[:, :6 - l], y[:, 6 - l:] if l else None, V[:, 6:], 0, False)[0] for l in range(6)}
            est.append(ap.select(J, 6, 6, T)["bic2"])
        return np.mean(est), mode(np.array(est, float))

    def compare(name, A, B):
        ca, cb = ap.cca(A[:, :6], A[:, 6:]), ap.cca(B[:, :6], B[:, 6:])
        (ma, moa), (mb, mob) = bic2(A), bic2(B)
        pool = np.vstack([A, B])
        g = []
        for _ in range(n_draw):
            i = rng.permutation(len(pool))
            a, b = pool[i[:len(A)]], pool[i[len(A):]]
            g.append(np.append(ap.cca(a[:, :6], a[:, 6:])[:3] - ap.cca(b[:, :6], b[:, 6:])[:3], bic2(a)[0] - bic2(b)[0]))
        g = np.array(g)
        obs = np.append(ca[:3] - cb[:3], ma - mb)
        pv = (1 + (g >= obs - 1e-12).sum(0)) / (1 + n_draw)
        return [f"{name}: T {len(A)} against {len(B)}",
                f"   canonical correlations 1 to 3: {ca[0]:.3f} {ca[1]:.3f} {ca[2]:.3f} against {cb[0]:.3f} {cb[1]:.3f} {cb[2]:.3f}   "
                f"P {pv[0]:.3f} {pv[1]:.3f} {pv[2]:.3f}",
                f"   r by BIC2 (most frequent over 30 orders): {int(moa)} against {int(mob)}; mean {ma:.2f} against {mb:.2f}   P {pv[3]:.3f}"]

    lines = ["Robustness of the two-market comparison (FUT6 and UST6, each event scaled inside its own group).",
             f"A draw mixes the scaled events of the two groups and keeps the group sizes. {n_draw} draws. One-sided P: the first group has more.", ""]
    lines += compare("Statements from 2001 against minutes (same years)", get("Statements", "2001-01-01"), get("Minutes", "2001-01-01"))
    lines += compare("Statements from 2011 against minutes from 2011", get("Statements", "2011-04-01"), get("Minutes", "2011-04-01"))
    lines += compare("Press conferences against minutes from 2011", get("Press Conferences", "2011-04-01"), get("Minutes", "2011-04-01"))
    lines += compare("Statements against minutes, roles exchanged (Treasury yields as responses)",
                     get("Statements", "1994-01-01", True), get("Minutes", "1994-01-01", True))
    return lines


if __name__ == "__main__":
    part = sys.argv[1].upper()
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if part == "J":
        arg = arg or "independent"
        lines = part_j(arg)
    else:
        lines = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f, "G": part_g, "H": part_h,
                 "I": part_i, "K": part_k, "L": part_l}[part]()
    name = f"review_checks_{part}.txt" if part != "J" else f"review_checks_J_{arg}.txt"
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(f"Review checks of 19 Nov 2026, part {part}. Script: code/review_checks.py.\n"
                 "Notation: m = all factors, r = factors or effects, L = the number used in a test. k is not used.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
