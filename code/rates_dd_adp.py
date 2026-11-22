"""The merged rates panel (USMPD + TAQ ETFs): AH (2013) with double demeaning, and Ahn and Perez (2010) / ADP (2018)
counts on small sub-panels (17 Nov 2026).
Output: results/rates_merge_2026-11-17/rates_dd_adp.txt. Data and loading as in rates_big_panel.py.

Part 1. AH with double demeaning. Double demeaning (AH 2013, eqs. 6 and 7, footnote 3) takes out the mean of each
  series and the cross-sectional mean of each event. That mean is only meaningful when all series move the same way
  for the same news. So the signs are aligned first: bond ETF price returns, EURUSD and FXE are multiplied by -1, so that
  every series rises with yields or with the dollar. Then each series is standardized (different units), optionally each
  event is scaled to equal size, then double demeaned. The cross-sectional mean of an event is then close to the
  "level" move of all rates, so double demeaning takes out the mean loading of the dominant factor, as it takes out the
  mean market beta for stocks.
Part 2. Ahn and Perez (2010) m and ADP (2018) r with ap_tools_v2.py. N is kept at 8 to 10 (more series give too many
  moments for T). Each event scaled to equal size. 100 random splits. The criteria cannot exceed Q - 1.
Part 3. The same ADP count on TAQ ETFs only, statement days against control days (credit and FX ETFs on Treasury ETFs).
Notation: m = all factors, r = factors tied to candidates. k is not used (in the lecture notes k counts regressors).

Usage: python3 rates_dd_adp.py
"""
import os

import numpy as np
import pandas as pd

import ap_tools_v2 as ap
import paths
from ap_tools import table
from factor_count_v2 import ahn_horenstein
from rates_big_panel import E_GROUPS, OUT, U_GROUPS, ah_line, load, scale_rows, std

FLIP = {"EURUSD", "E_FXE", "E_BIL", "E_SHV", "E_SHY", "E_IEI", "E_IEF", "E_TLT", "E_TIP",
        "E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB"}


def fmt(res, keys=("bic2", "seq", "bic1")):
    return "  ".join(f"{k} {table(res[k])}" for k in keys)


def dd(Z):
    return Z - Z.mean(1, keepdims=True) - Z.mean(0, keepdims=True) + Z.mean()


def main():
    rng = np.random.default_rng(20261118)
    E, u, st, A, B = load()
    ucols = [c for g in U_GROUPS.values() for c in g]
    ecols = [c for g in E_GROUPS.values() for c in g]
    days = [d for d in st if d >= "2008-01-01" and d in u.index and d in E.index
            and u.loc[d, ucols].notna().all() and E.loc[d, ecols].notna().all()]
    D = pd.concat([u.loc[days, ucols].astype(float), E.loc[days, ecols].astype(float)], axis=1)
    S = D.copy()
    for c in S.columns:
        if c in FLIP:
            S[c] = -S[c]
    lines = [__doc__.split("Usage")[0].strip(), "", f"Statement days in both sources, 2008 to 2025: {len(days)}.", ""]

    lines.append("Part 1. AH (2013) with signs aligned and double demeaning.")
    Z0 = std(S.values)
    U_, s_, Vt = np.linalg.svd(Z0 - Z0.mean(0), full_matrices=False)
    neg = [c for c, v in zip(S.columns, Vt[0] * np.sign(Vt[0].sum())) if v < 0]
    lines.append(f"   Check: series with a negative loading on the first component after the sign alignment: {neg or 'none'}")
    for name, cols in {"USMPD only (27)": ucols, "TAQ rate ETFs only (16)": ecols, "USMPD + TAQ (43)": ucols + ecols}.items():
        Z = std(S[cols].values)
        lines.append(ah_line(Z, name + ", standardized (no dd)"))
        lines.append(ah_line(dd(Z), name + ", standardized, dd"))
        lines.append(ah_line(dd(scale_rows(Z)), name + ", standardized, scaled, dd"))
    lines.append("")

    lines.append("Part 2. Ahn and Perez (2010) and ADP (2018) on small sub-panels of the merged data (statement days).")
    X = pd.DataFrame(scale_rows(std(D.values)), index=D.index, columns=D.columns)
    rep = ["MP1", "ED4", "UST2Y", "UST10Y", "UST30Y", "TIPS10Y", "E_LQD", "E_HYG", "E_MBB", "DXY"]
    m = ap.count_m(X[rep].values, 100, 0, rng)
    lines.append(f"   m, one series per market (MP1 ED4 UST2Y UST10Y UST30Y TIPS10Y LQD HYG MBB DXY), N 10: " + fmt(m))
    cases = [("credit and FX ETFs (AGG BND HYG JNK LQD MBB MUB FXE UUP)", ["E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB", "E_FXE", "E_UUP"],
              "USMPD (MP1 ED4 UST2Y UST10Y TIPS10Y)", ["MP1", "ED4", "UST2Y", "UST10Y", "TIPS10Y"]),
             ("USMPD (MP1 MP2 ED4 ED8 UST2Y UST10Y UST30Y TIPS10Y)", ["MP1", "MP2", "ED4", "ED8", "UST2Y", "UST10Y", "UST30Y", "TIPS10Y"],
              "TAQ (SHY IEF TLT TIP LQD HYG)", ["E_SHY", "E_IEF", "E_TLT", "E_TIP", "E_LQD", "E_HYG"]),
             ("credit and FX ETFs", ["E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB", "E_FXE", "E_UUP"],
              "Swanson-type dimensions (MP1 ED4 UST10Y)", ["MP1", "ED4", "UST10Y"])]
    for yname, ycols, zname, zcols in cases:
        Y, Zc = X[ycols].values, X[zcols].values
        mm = ap.count_m(Y, 100, 0, rng)
        rr = ap.count_r(Y, Zc, 100, 0, rng)
        lines.append(f"   Y = {yname}, N {len(ycols)}")
        lines.append(f"      m of Y: " + fmt(mm))
        lines.append(f"      r on Z = {zname}, Q {len(zcols)}: " + fmt(rr))
    lines.append("")

    lines.append("Part 3. TAQ ETFs only: credit and FX ETFs (Y) on Treasury and TIPS ETFs (Z = SHY IEF TLT TIP), statement vs control days.")
    yc = ["E_AGG", "E_BND", "E_HYG", "E_JNK", "E_LQD", "E_MBB", "E_MUB", "E_FXE", "E_UUP"]
    zc = ["E_SHY", "E_IEF", "E_TLT", "E_TIP"]
    good = lambda d: d in E.index and E.loc[d, yc + zc].notna().all()
    st8 = [d for d in st if d >= "2008-01-01" and good(d)]
    for sname, ds in {"statement days": st8, "control, one week before": [A[d] for d in st8 if d in A and good(A[d])],
                      "control, one week after": [B[d] for d in st8 if d in B and good(B[d])]}.items():
        V = scale_rows(std(E.loc[ds, yc + zc].astype(float).values))
        mm = ap.count_m(V[:, :len(yc)], 100, 0, rng)
        rr = ap.count_r(V[:, :len(yc)], V[:, len(yc):], 100, 0, rng)
        lines.append(f"   {sname:25s} T {len(ds)}:  m of Y: {fmt(mm, ('bic2', 'seq'))}   r: {fmt(rr, ('bic2', 'seq'))}")
    with open(os.path.join(OUT, "rates_dd_adp.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("Part 1. AH (2013) with signs aligned and double demeaning.") - 1:]))


if __name__ == "__main__":
    main()
