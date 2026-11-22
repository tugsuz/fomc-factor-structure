"""Monthly factor counts on Fama-French portfolios (10 Nov 2026). A test of the idea
("try monthly data; size and value factors sometimes show up").

Data. If Ken French's monthly files are in data_public/french_monthly/ (download_french_monthly.sh),
they are used. If not, monthly returns are built from the daily files in data_public/french/ by
compounding the daily returns inside each calendar month. The daily files start in July 1926
(size and book-to-market, industries), November 1926 (momentum) and July 1963 (profitability,
investment), so the long samples are possible either way. Monthly excess returns subtract the
compounded daily T-bill return. A month with a missing daily return is missing.

Panels (value-weighted portfolios):
  25 size-BM; 100 size-BM; 49 industries; 25 size-BM plus 49 industries;
  25 size-BM plus 25 size-momentum; the 379 panel of the daily work (100 size-BM, 100 size-OP,
  100 size-INV, 25 size-momentum, 10 momentum, 49 industries; columns complete in the sample);
  and, when the monthly files are there, the four 25-portfolio sorts on size together.
Samples: from July 1926 (where the panel exists), from July 1963, from January 1994, and
July 1963 to December 1993.
Versions of each panel (all counted with factor_count_v2.ahn_horenstein, kmax 8, r = 0 allowed):
  firm    each series demeaned over time; the market stays in
  double  double demeaning, the paper's recommendation
  exmkt   each series minus the market return (unit betas), then demeaned over time
          (the variable of Ahn and Horenstein 2017, "excess returns over the market return")
  mktout  residuals of a regression of each series on the market excess return, then double demeaning
The same table is made for daily returns, so that the two frequencies can be compared.

Factor identity. For each panel and version, the first five principal components are compared with
the Fama-French factors: the R squared of SMB and of HML on the first k components (k = 1, 2, 3, 5)
and the canonical correlations between the first three components and (Mkt-RF, SMB, HML).

Run:  python3 monthly_counts.py      Output: results/monthly_counts_2026-11-10/
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

import paths
from factor_count_v2 import ahn_horenstein, bai_ng, double_demean, onatski_ed

OUT = os.path.join(paths.RESULTS, "monthly_counts_2026-11-10")
MONTHLY = os.path.join(paths.DATA, "french_monthly")
FAMILIES = {   # tag: (monthly file, daily file)
    "BM25": ("25_Portfolios_5x5_CSV.zip", "25_Portfolios_5x5_Daily_CSV.zip"),
    "BM100": ("100_Portfolios_10x10_CSV.zip", "100_Portfolios_10x10_Daily_CSV.zip"),
    "OP100": ("100_Portfolios_ME_OP_10x10_CSV.zip", "100_Portfolios_ME_OP_10x10_daily_CSV.zip"),
    "INV100": ("100_Portfolios_ME_INV_10x10_CSV.zip", "100_Portfolios_ME_INV_10x10_daily_CSV.zip"),
    "SZMOM25": ("25_Portfolios_ME_Prior_12_2_CSV.zip", "25_Portfolios_ME_Prior_12_2_Daily_CSV.zip"),
    "MOM10": ("10_Portfolios_Prior_12_2_CSV.zip", "10_Portfolios_Prior_12_2_Daily_CSV.zip"),
    "IND49": ("49_Industry_Portfolios_CSV.zip", "49_Industry_Portfolios_daily_CSV.zip"),
    "OP25": ("25_Portfolios_ME_OP_5x5_CSV.zip", None),
    "INV25": ("25_Portfolios_ME_INV_5x5_CSV.zip", None),
}
SAMPLES = [("1926-07", "2026-12"), ("1963-07", "2026-12"), ("1994-01", "2026-12"), ("1963-07", "1993-12")]
VERSIONS = ["firm", "double", "exmkt", "mktout"]


def first_block(zip_path):
    """First data block of a French CSV (value-weighted returns, or the factors), in percent.
    Daily files have dates YYYYMMDD, monthly files YYYYMM. Missing codes become NaN."""
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(","))
    block = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        block.append(l)
    df = pd.read_csv(io.StringIO("\n".join(block)), skipinitialspace=True)
    df = df.rename(columns={df.columns[0]: "date"})
    s = df["date"].astype(str).str.strip()
    if s.str.len().iloc[0] == 8:
        df["date"] = pd.to_datetime(s, format="%Y%m%d")
    else:
        df["date"] = pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").astype(float)
    df[df <= -99.99] = np.nan
    df.columns = [c.strip() for c in df.columns]
    return df


def compound(d):
    """Daily percent returns to monthly percent returns. A month with a missing day is missing."""
    g = (1 + d / 100).groupby(d.index.to_period("M"))
    m = (g.prod(min_count=1) - 1) * 100
    m[g.count() < g.size().values[:, None]] = np.nan
    m.index = m.index.to_timestamp("M")
    return m


def load_all():
    """Returns (daily dict, monthly dict, daily factors, monthly factors, note on the source)."""
    dly, mly, src = {}, {}, {}
    for tag, (fm, fd) in FAMILIES.items():
        if fd and os.path.exists(os.path.join(paths.FRENCH, fd)):
            dly[tag] = first_block(os.path.join(paths.FRENCH, fd))
        if os.path.exists(os.path.join(MONTHLY, fm)):
            mly[tag], src[tag] = first_block(os.path.join(MONTHLY, fm)), "monthly file"
        elif tag in dly:
            mly[tag], src[tag] = compound(dly[tag]), "compounded from daily"
    f3 = first_block(os.path.join(paths.FRENCH, "F-F_Research_Data_Factors_daily_CSV.zip"))
    f5 = first_block(os.path.join(paths.FRENCH, "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"))
    mom = first_block(os.path.join(paths.FRENCH, "F-F_Momentum_Factor_daily_CSV.zip"))
    fd = f3.join(f5[["RMW", "CMA"]]).join(mom.rename(columns={mom.columns[0]: "MOM"}))
    if os.path.exists(os.path.join(MONTHLY, "F-F_Research_Data_Factors_CSV.zip")):
        fm = first_block(os.path.join(MONTHLY, "F-F_Research_Data_Factors_CSV.zip"))
        for f, cols in [("F-F_Research_Data_5_Factors_2x3_CSV.zip", ["RMW", "CMA"]), ("F-F_Momentum_Factor_CSV.zip", None)]:
            p = os.path.join(MONTHLY, f)
            if os.path.exists(p):
                x = first_block(p)
                fm = fm.join(x[cols] if cols else x.rename(columns={x.columns[0]: "MOM"}))
        src["factors"] = "monthly file"
    else:
        tot = compound(fd[["Mkt-RF"]].add(fd["RF"], axis=0))["Mkt-RF"]
        rf = compound(fd[["RF"]])["RF"]
        fm = compound(fd[[c for c in fd.columns if c not in ("Mkt-RF", "RF")]])
        fm.insert(0, "Mkt-RF", tot - rf)
        fm["RF"] = rf
        src["factors"] = "compounded from daily (SMB, HML and the other spreads are approximate)"
    return dly, mly, fd, fm, src


def panels(fam):
    def cat(tags):
        if not all(t in fam for t in tags):
            return None
        return pd.concat([fam[t].add_prefix(t + ":") for t in tags], axis=1)
    p = {"25 size-BM": cat(["BM25"]), "100 size-BM": cat(["BM100"]), "49 industries": cat(["IND49"]),
         "25 size-BM + 49 industries": cat(["BM25", "IND49"]), "25 size-BM + 25 size-momentum": cat(["BM25", "SZMOM25"]),
         "379 panel (as daily work)": cat(["BM100", "OP100", "INV100", "SZMOM25", "MOM10", "IND49"]),
         "4 x 25 size sorts (BM, OP, INV, momentum)": cat(["BM25", "OP25", "INV25", "SZMOM25"])}
    return {k: v for k, v in p.items() if v is not None}


def prep(X, mkt, v):
    """X: excess returns (T x N); mkt: market excess return (T). Returns the panel and the demean flag."""
    if v == "firm":
        return X - X.mean(0, keepdims=True), False
    if v == "double":
        return X, True
    if v == "exmkt":
        Z = X - mkt[:, None]
        return Z - Z.mean(0, keepdims=True), False
    A = np.column_stack([np.ones(len(mkt)), mkt])
    return X - A @ np.linalg.lstsq(A, X, rcond=None)[0], True


def cancor(a, b):
    qa, _ = np.linalg.qr(a - a.mean(0))
    qb, _ = np.linalg.qr(b - b.mean(0))
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def r2(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    e = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()


def analyse(R, F, name, freq, start, end, rows, lines):
    R = R.loc[start:end]
    full = R.notna().all()
    if full.mean() < 0.6:            # a family of the panel starts later than this sample: skip it
        return
    R = R.loc[:, full]
    idx = R.index.intersection(F.dropna(subset=["Mkt-RF", "SMB", "HML", "RF"]).index)
    if len(idx) < 60 or R.shape[1] < 20:
        return
    R, f = R.loc[idx], F.loc[idx]
    X = R.values - f["RF"].values[:, None]
    mkt = f["Mkt-RF"].values
    T, N = X.shape
    lines.append(f"{name} | {freq} | {idx[0]:%Y-%m} to {idx[-1]:%Y-%m} | T = {T}, N = {N}")
    for v in VERSIONS:
        Z, dd = prep(X, mkt, v)
        o = ahn_horenstein(Z, kmax=8, demean=dd)
        Zr = double_demean(Z) if dd else Z
        U, S, _ = np.linalg.svd(Zr, full_matrices=False)
        pcs = U[:, :5] * S[:5]
        ff = f[["Mkt-RF", "SMB", "HML"]].values
        cc = cancor(pcs[:, :3], ff)
        rs = {k: r2(f["SMB"].values, pcs[:, :k]) for k in (1, 2, 3, 5)}
        rh = {k: r2(f["HML"].values, pcs[:, :k]) for k in (1, 2, 3, 5)}
        rm = {k: r2(mkt, pcs[:, :k]) for k in (1, 2, 3)}
        ic, bic = bai_ng(Z, kmax=8, demean=dd)
        ed = onatski_ed(Z, rmax=8, demean=dd)
        er, sh = o["er"], o["shares"]
        lines.append(f"   {v:6s} r_ER {o['r_er']} r_GR {o['r_gr']} | ER(1..3) {er[1]:.2f} {er[2]:.2f} {er[3]:.2f} | shares {sh[0]:.3f} {sh[1]:.3f} {sh[2]:.3f} "
                     f"| switch {o['switch']:.2f} | BIC3 {bic} ED {ed} | R2 of SMB on first 1, 2, 3, 5 PCs: {rs[1]:.2f} {rs[2]:.2f} {rs[3]:.2f} {rs[5]:.2f}; "
                     f"of HML: {rh[1]:.2f} {rh[2]:.2f} {rh[3]:.2f} {rh[5]:.2f}; of the market on 1, 2, 3: {rm[1]:.2f} {rm[2]:.2f} {rm[3]:.2f} "
                     f"| canonical corr. PC1-3 with (Mkt, SMB, HML): {cc[0]:.2f} {cc[1]:.2f} {cc[2]:.2f}")
        rows.append({"panel": name, "freq": freq, "start": f"{idx[0]:%Y-%m}", "end": f"{idx[-1]:%Y-%m}", "T": T, "N": N, "version": v,
                     "r_ER": o["r_er"], "r_GR": o["r_gr"], "ER1": er[1], "ER2": er[2], "ER3": er[3], "s1": sh[0], "s2": sh[1], "s3": sh[2],
                     "switch": o["switch"], "BIC3": bic, "ED": ed, "R2_SMB_1": rs[1], "R2_SMB_2": rs[2], "R2_SMB_3": rs[3], "R2_SMB_5": rs[5],
                     "R2_HML_1": rh[1], "R2_HML_2": rh[2], "R2_HML_3": rh[3], "R2_HML_5": rh[5], "R2_MKT_1": rm[1],
                     "cc1": cc[0], "cc2": cc[1], "cc3": cc[2]})
    lines.append("")


def main():
    os.makedirs(OUT, exist_ok=True)
    dly, mly, fd, fm, src = load_all()
    lines = ["Monthly and daily factor counts on Fama-French portfolios (monthly_counts.py).",
             "Source of the monthly returns: " + "; ".join(f"{k} {v}" for k, v in src.items()), ""]
    rows = []
    for freq, fam, F in [("monthly", mly, fm), ("daily", dly, fd)]:
        for name, R in panels(fam).items():
            for start, end in SAMPLES:
                analyse(R, F, name, freq, start, end, rows, lines)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "monthly_counts.csv"), index=False)
    open(os.path.join(OUT, "monthly_counts.txt"), "w").write("\n".join(lines) + "\n")
    t["count"] = t["r_ER"].astype(str) + "/" + t["r_GR"].astype(str)
    t["sample"] = t["start"] + " to " + t["end"]
    wide = t.pivot_table(index=["panel", "sample", "N"], columns=["freq", "version"], values="count", aggfunc="first")
    wide = wide.reindex(columns=[(f, v) for f in ("monthly", "daily") for v in VERSIONS])
    summ = "Counts r_ER/r_GR by panel, sample, frequency and version\n" + wide.to_string()
    open(os.path.join(OUT, "monthly_counts_summary.txt"), "w").write(summ + "\n")
    print(summ)


if __name__ == "__main__":
    main()
