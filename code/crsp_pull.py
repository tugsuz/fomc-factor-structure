"""CRSP daily stock returns from WRDS for the stock panels of Result 1.

What it pulls: one file per calendar year, data_wrds/crsp_daily_<year>.csv.gz, with the columns
  permno, date, ret (daily return in percent), prc (price, absolute value; CRSP marks bid-ask
  averages with a minus sign), cap (market capitalization in millions of dollars), vol (volume)
for every common share of a U.S. company listed on NYSE, AMEX or Nasdaq, from the CRSP daily
stock file version 2 (crsp.dsf_v2) joined to the security-information history
(crsp.stksecurityinfohist) with the filters that correspond to the legacy share codes 10 and 11
and exchange codes 1, 2, 3:
  sharetype = 'NS', securitytype = 'EQTY', securitysubtype = 'COM', usincflg = 'Y',
  issuertype in ('ACOR', 'CORP'), primaryexch in ('N', 'A', 'Q'),
  conditionaltype in ('RW', 'NW'), tradingstatusflg = 'A'.
(The join and the filters follow the WRDS version-2 documentation as used in the Tidy Finance
CRSP chapter, tidy-finance.org/chapters/wrds-crsp-and-compustat.html; checked 28 Oct 2026.)

How to run (needs the wrds package, pip install wrds, and a WRDS account; the connection asks
for the username and password, or reads ~/.pgpass if you have set one up):
  python crsp_pull.py --probe                  prints the columns of the two tables and the row
                                               count of one month, so that the column names can be
                                               checked before the full pull (dlycap and shrout are
                                               taken if present, otherwise cap = |prc| x shrout)
  python crsp_pull.py --start 1994 --end 2026  the pull, one year at a time; years whose file
                                               already exists are skipped (restart after a dropped
                                               connection by running the same command again)
  python crsp_pull.py --user yourname ...      to pass the WRDS username on the command line
  python crsp_pull.py --out /some/folder ...   to write the files somewhere else than data_wrds/
Outputs go to data_wrds/ (created if missing); the SQL and the row counts are logged
in data_wrds/pull_log.txt. Expect about 1.5 to 2.5 million rows per year and 20 to 30 MB per file.
"""
import argparse
import datetime as dt
import gzip
import os
import sys
import time

import pandas as pd

import paths

OUT_DIR = os.path.join(paths.ROOT, "data_wrds")
FILTERS = ("s.sharetype = 'NS' AND s.securitytype = 'EQTY' AND s.securitysubtype = 'COM' "
           "AND s.usincflg = 'Y' AND s.issuertype IN ('ACOR', 'CORP') "
           "AND s.primaryexch IN ('N', 'A', 'Q') AND s.conditionaltype IN ('RW', 'NW') AND s.tradingstatusflg = 'A'")
JOIN = ("FROM crsp.dsf_v2 AS d JOIN crsp.stksecurityinfohist AS s "
        "ON d.permno = s.permno AND s.secinfostartdt <= d.dlycaldt AND d.dlycaldt <= s.secinfoenddt")


def connect(user):
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: pip install wrds")
    return wrds.Connection(wrds_username=user) if user else wrds.Connection()


def columns(db, table):
    q = ("SELECT column_name FROM information_schema.columns "
         f"WHERE table_schema = 'crsp' AND table_name = '{table}' ORDER BY ordinal_position")
    return db.raw_sql(q)["column_name"].tolist()


def select_list(dsf_cols):
    parts = ["d.permno", "d.dlycaldt AS date", "d.dlyret AS ret", "d.dlyprc AS prc"]
    if "dlycap" in dsf_cols:
        parts.append("d.dlycap AS cap_raw")
    if "shrout" in dsf_cols:
        parts.append("d.shrout AS shrout")
    if "dlyvol" in dsf_cols:
        parts.append("d.dlyvol AS vol")
    return ", ".join(parts)


def tidy(df, dsf_cols):
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = 100.0 * pd.to_numeric(df["ret"], errors="coerce")
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
    if "cap_raw" in df.columns:
        df["cap"] = pd.to_numeric(df["cap_raw"], errors="coerce") / 1000.0      # dlycap is in thousands of dollars
    elif "shrout" in df.columns:
        df["cap"] = df["prc"] * pd.to_numeric(df["shrout"], errors="coerce") / 1000.0   # shrout in thousands
    else:
        df["cap"] = float("nan")
    keep = ["permno", "date", "ret", "prc", "cap"] + (["vol"] if "vol" in df.columns else [])
    out = df[keep].sort_values(["date", "permno"]).reset_index(drop=True)
    return out


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--start", type=int, default=1994)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=OUT_DIR, help="output folder (default data_wrds)")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    db = connect(args.user)
    dsf_cols = columns(db, "dsf_v2")
    sih_cols = columns(db, "stksecurityinfohist")
    log = open(os.path.join(OUT_DIR, "pull_log.txt"), "a")
    log.write(f"\n== {dt.datetime.now().isoformat(timespec='seconds')}\n")
    if args.probe:
        print("crsp.dsf_v2 columns:", dsf_cols)
        print("crsp.stksecurityinfohist columns:", sih_cols)
        need = ["dlycaldt", "dlyret", "dlyprc"]
        print("required present:", all(c in dsf_cols for c in need), "; cap from:",
              "dlycap" if "dlycap" in dsf_cols else ("prc x shrout" if "shrout" in dsf_cols else "NOT AVAILABLE"))
        q = (f"SELECT COUNT(*) AS n, COUNT(DISTINCT d.permno) AS n_permno {JOIN} "
             f"WHERE d.dlycaldt BETWEEN '2024-06-01' AND '2024-06-30' AND {FILTERS}")
        print("June 2024 after the filters:", db.raw_sql(q).to_dict("records"))
        q2 = f"SELECT {select_list(dsf_cols)} {JOIN} WHERE d.dlycaldt = '2024-06-12' AND {FILTERS} LIMIT 5"
        print(db.raw_sql(q2))
        return
    sel = select_list(dsf_cols)
    for year in range(args.start, args.end + 1):
        path = os.path.join(OUT_DIR, f"crsp_daily_{year}.csv.gz")
        if os.path.exists(path) and not args.force:
            print(f"{year}: exists, skipped")
            continue
        q = f"SELECT {sel} {JOIN} WHERE d.dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31' AND {FILTERS}"
        t0 = time.time()
        df = db.raw_sql(q)
        out = tidy(df, dsf_cols)
        tmp = path + ".part"
        with gzip.open(tmp, "wt") as f:
            out.to_csv(f, index=False, float_format="%.6g")
        os.replace(tmp, path)
        msg = f"{year}: {len(out)} rows, {out.permno.nunique()} permnos, {out.date.nunique()} days, {time.time() - t0:.0f} s"
        print(msg)
        log.write(msg + "\n")
        log.flush()
    log.write("SQL template: SELECT " + sel + " " + JOIN + " WHERE d.dlycaldt BETWEEN <year> AND " + FILTERS + "\n")
    log.close()


if __name__ == "__main__":
    main()
