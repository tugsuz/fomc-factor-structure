"""A larger rates panel: USMPD series and TAQ ETFs together, with Ahn and Horenstein (2013) and Ahn and Zhang (2023)
(17 Nov 2026). Output: results/rates_merge_2026-11-17/.

Data. Scheduled FOMC statements from 2008 to 2025 that are in both sources. One row per statement: the USMPD window
changes (10 minutes before to 20 minutes after) and the TAQ ETF returns in the same window (w30).
  USMPD (27): MP1 MP2 FF3-FF6 ED1-ED8 OIS1Y, UST3M UST6M UST2Y UST5Y UST10Y UST30Y, TIPS5Y TIPS10Y TIPS30Y, DXY EURUSD USDJPY.
  TAQ ETFs (16): BIL SHV SHY IEI IEF TLT, TIP, AGG BND HYG JNK LQD MBB MUB, FXE UUP.
  With equities (+9): USMPD SP500 SPFUT; TAQ SPY XLF KBE KRE XLU IYR VNQ.
The units differ (yields in percentage points, ETF returns in percent), so each series is standardized over the events.
Version "scaled": then each event is divided by its root mean square, so that a few huge events (2008, 2020) do not
decide the count (as in v2_ap_rerun.py). Double demeaning is not used: the series have different units and signs
(an ETF price falls when its yield rises).

Tools.
  AH     factor_count_v2.ahn_horenstein (ER and GR, kmax 8, mock eigenvalue V(0)/ln m), demean=False after the
         standardization above.
  AZ     Ahn and Zhang (2023) Algorithm 1, multilevel_tools.algorithm1: global factors (shared by every
         group), regional factors (shared by some groups), local factors (one group). Groups are the markets.
  Control days exist only for the TAQ ETFs. Part 3 compares statement days with their control days (same clock time,
  one week before and after) for the 16 rate ETFs, with the matched-block permutation (one day from each block).
Notation: r = number of factors. k is not used (in the panel lecture notes k counts regressors).

Usage: python3 rates_big_panel.py
"""
import os

import numpy as np
import pandas as pd

import multilevel_tools as ML
import paths
from factor_count_v2 import ahn_horenstein

OUT = os.path.join(paths.RESULTS, "rates_merge_2026-11-17")
os.makedirs(OUT, exist_ok=True)
U_GROUPS = {"money market": ["MP1", "MP2", "FF3", "FF4", "FF5", "FF6", "ED1", "ED2", "ED3", "ED4", "ED5", "ED6", "ED7", "ED8", "OIS1Y"],
            "treasury": ["UST3M", "UST6M", "UST2Y", "UST5Y", "UST10Y", "UST30Y"],
            "tips": ["TIPS5Y", "TIPS10Y", "TIPS30Y"],
            "credit": [],
            "fx": ["DXY", "EURUSD", "USDJPY"]}
E_GROUPS = {"money market": [], "treasury": ["E_BIL", "E_SHV", "E_SHY", "E_IEI", "E_IEF", "E_TLT"], "tips": ["E_TIP"],
            "credit": ["E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB"], "fx": ["E_FXE", "E_UUP"]}
EQ_U, EQ_E = ["SP500", "SPFUT"], ["E_SPY", "E_XLF", "E_KBE", "E_KRE", "E_XLU", "E_IYR", "E_VNQ"]
SEED = 20261117


def std(X):
    return (X - X.mean(0)) / X.std(0, ddof=1)


def scale_rows(Z):
    return Z / np.sqrt((Z ** 2).mean(1, keepdims=True))


def ah_line(Z, tag):
    o = ahn_horenstein(Z, kmax=8, demean=False)
    sh = o["eigenvalues"][:5] / o["eigenvalues"].sum()
    return (f"   {tag:44s} T {Z.shape[0]} N {Z.shape[1]}: ER {o['r_er']}  GR {o['r_gr']}  "
            f"ER(0..3) {' '.join(f'{x:.2f}' for x in o['er'][:4])}  eigenvalue shares {' '.join(f'{x:.3f}' for x in sh)}")


def load():
    w = pd.read_csv(os.path.join(paths.ROOT, "data_wrds", "taq", "windows_all.csv.gz"), usecols=["date", "kind", "id", "w30"])
    E = w[w["kind"] == "etf"].pivot_table(index="date", columns="id", values="w30", aggfunc="first")
    t = pd.read_csv(os.path.join(paths.ROOT, "data_wrds", "taq", "targets.csv"), dtype=str)
    st = sorted(t[(t["set"] == "statement") & (t["unscheduled"] == "0")]["date"])
    c = t[t["set"] == "control"]
    A = {p: x for x, p in zip(c["date"], c["pair"]) if x < p}
    B = {p: x for x, p in zip(c["date"], c["pair"]) if x > p}
    u = pd.read_excel(os.path.join(paths.FED, "USMPD.xlsx"), "Statements")
    u = u[u["Unscheduled"] == 0].copy()
    u["date"] = pd.to_datetime(u["Date"]).dt.strftime("%Y-%m-%d")
    return E, u.set_index("date"), st, A, B


def main():
    E, u, st, A, B = load()
    ucols = [c for g in U_GROUPS.values() for c in g]
    ecols = [c for g in E_GROUPS.values() for c in g]
    days = [d for d in st if d >= "2008-01-01" and d in u.index and d in E.index
            and u.loc[d, ucols + EQ_U].notna().all() and E.loc[d, ecols + EQ_E].notna().all()]
    D = pd.concat([u.loc[days, ucols + EQ_U].astype(float), E.loc[days, ecols + EQ_E].astype(float)], axis=1)
    lines = [__doc__.split("Usage")[0].strip(), "", f"Statement days in both sources, 2008 to 2025: {len(days)}.", ""]

    lines.append("Part 1. Ahn and Horenstein (2013) on statement days.")
    panels = {"USMPD only (27)": ucols, "TAQ rate ETFs only (16)": ecols, "USMPD + TAQ, rates (43)": ucols + ecols,
              "USMPD + TAQ, rates and equities (52)": ucols + ecols + EQ_U + EQ_E}
    for name, cols in panels.items():
        Z = std(D[cols].values)
        lines.append(ah_line(Z, name + ", standardized"))
        lines.append(ah_line(scale_rows(Z), name + ", standardized and scaled"))
    lines.append("   Dominant first factor check (my own step, not in the paper): take out the first k principal components, then ER and GR again.")
    for name in ["USMPD only (27)", "USMPD + TAQ, rates (43)"]:
        Z = scale_rows(std(D[panels[name]].values))
        Uz, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        for k_out in (1, 2):
            Rz = Z - Z.mean(0) - (Uz[:, :k_out] * s[:k_out]) @ Vt[:k_out]
            lines.append(ah_line(Rz, f"{name}, scaled, first {k_out} PC out"))
    for a, b in [("2008-01-01", "2015-12-31"), ("2016-01-01", "2025-12-31")]:
        sub = D[(D.index >= a) & (D.index <= b)]
        Z = scale_rows(std(sub[ucols + ecols].values))
        lines.append(ah_line(Z, f"USMPD + TAQ, rates (43), scaled, {a[:4]}-{b[:4]}"))
    lines.append("")

    lines.append("Part 2. Ahn and Zhang (2023), Algorithm 1. Groups = markets. Data standardized and scaled (rates panel of 43).")
    for with_eq in (False, True):
        names = list(U_GROUPS) + (["equity"] if with_eq else [])
        cols_g = [U_GROUPS[g] + E_GROUPS[g] for g in U_GROUPS] + ([EQ_U + EQ_E] if with_eq else [])
        allc = [c for g in cols_g for c in g]
        Z = pd.DataFrame(scale_rows(std(D[allc].values)), index=D.index, columns=allc)
        blocks = [Z[g].values for g in cols_g]
        res = ML.algorithm1(blocks)
        lines.append(f"   groups: " + ", ".join(f"{n} ({len(g)})" for n, g in zip(names, cols_g)))
        lines += ["   " + x for x in ML.describe(res, names)]
        vs = ML.variance_shares(blocks, res)
        lines.append("   variance shares (global / regional / local / idiosyncratic): " +
                     "; ".join(f"{n} {v['global']:.2f} / {v['regional']:.2f} / {v['local']:.2f} / {v['idiosyncratic']:.2f}" for n, v in zip(names, vs)))
        lines.append("")

    lines.append("Part 3. The 16 TAQ rate ETFs: statement days against their control days. Matched-block permutation, 999 draws.")
    blocks = [[d] + [x for x in (A.get(d), B.get(d)) if x is not None and x in E.index and E.loc[x, ecols].notna().all()]
              for d in st if d >= "2008-01-01" and d in E.index and E.loc[d, ecols].notna().all()]
    blocks = [b for b in blocks if len(b) > 1]

    def stat(ds):
        Z = scale_rows(std(E.loc[ds, ecols].astype(float).values))
        o = ahn_horenstein(Z, kmax=6, demean=False)
        ev = o["eigenvalues"] / o["eigenvalues"].sum()
        return np.array([ev[0], ev[1], ev[:3].sum(), o["er"][1], o["r_er"]])

    rng = np.random.default_rng(SEED)
    obs = stat([b[0] for b in blocks])
    draws = np.array([stat([b[rng.integers(len(b))] for b in blocks]) for _ in range(999)])
    ctlA = stat([b[1] for b in blocks if b[1] < b[0]]) if all(len(b) > 1 for b in blocks) else None
    names = ["share 1", "share 2", "share 1-3", "ER(1)", "r by ER"]
    lines.append(f"   blocks: {len(blocks)}")
    for j, n in enumerate(names):
        p = (1 + np.sum(draws[:, j] >= obs[j] - 1e-12)) / 1000
        lines.append(f"   {n:10s} statement {obs[j]:.3f}  draws mean {draws[:, j].mean():.3f}  P {p:.3f}")
    with open(os.path.join(OUT, "rates_big_panel.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("Part 1. Ahn and Horenstein (2013) on statement days.") - 1:]))


if __name__ == "__main__":
    main()
