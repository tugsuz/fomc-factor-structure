"""CRSP daily Treasury file from WRDS, for a panel of U.S. Treasury securities (9 Nov 2026).

What it pulls, into data_wrds/treasury/:
  tfz_iss.csv.gz              the issue file (crsp.tfz_iss): one row per Treasury security, with its
                              type, coupon, dated date and maturity date (all columns)
  tfz_dly_<year>.csv.gz       the daily file (crsp.tfz_dly), all columns, one file per calendar year
The daily file has one row per security and trading day: prices, accrued interest, the daily
return, the yield to maturity, duration and amounts outstanding (the exact columns are printed by
--probe and kept as they are). A standard academic subscription can read both tables (checked 9 Nov 2026:
1961-06-14 to 2025-12-31, 6,466 securities, 609 traded in 2024).

How to run (the wrds package reads ~/.pgpass, so no password is asked):
  python3 crsp_treasury_pull.py --probe              prints the columns of both tables and one day's rows
  python3 crsp_treasury_pull.py                      the pull, 1994 to 2025, one year at a time; years whose
                                                     file already exists are skipped, so run the same
                                                     command again after a dropped connection
  python3 crsp_treasury_pull.py --start 1990 --end 2025   another range
Expect roughly 100,000 to 200,000 rows per year and a few MB per file. The log is
data_wrds/treasury/pull_log.txt.
"""
import argparse
import datetime as dt
import os
import sys
import time

import paths

OUT_DIR = os.path.join(paths.ROOT, "data_wrds", "treasury")
USER = os.environ.get("WRDS_USER", "")  # your WRDS user name


def connect():
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: python3 -m pip install wrds")
    return wrds.Connection(wrds_username=USER)


def log(msg):
    os.makedirs(OUT_DIR, exist_ok=True)
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with open(os.path.join(OUT_DIR, "pull_log.txt"), "a") as f:
        f.write(line + "\n")


def cols(db, table):
    q = ("SELECT column_name FROM information_schema.columns "
         f"WHERE table_schema = 'crsp' AND table_name = '{table}' ORDER BY ordinal_position")
    return db.raw_sql(q)["column_name"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--start", type=int, default=1994)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--force", action="store_true", help="pull years whose file already exists")
    a = ap.parse_args()
    db = connect()
    if a.probe:
        for t in ["tfz_iss", "tfz_dly"]:
            print(t, "columns:", cols(db, t))
        print(db.raw_sql("SELECT * FROM crsp.tfz_dly WHERE caldt = '2024-06-12' LIMIT 5").to_string())
        print(db.raw_sql("SELECT * FROM crsp.tfz_iss LIMIT 3").to_string())
        db.close()
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    f_iss = os.path.join(OUT_DIR, "tfz_iss.csv.gz")
    if a.force or not os.path.exists(f_iss):
        iss = db.raw_sql("SELECT * FROM crsp.tfz_iss")
        iss.to_csv(f_iss + ".part", index=False, compression="gzip")
        os.replace(f_iss + ".part", f_iss)
        log(f"tfz_iss: {len(iss)} rows")
    for y in range(a.start, a.end + 1):
        f = os.path.join(OUT_DIR, f"tfz_dly_{y}.csv.gz")
        if os.path.exists(f) and not a.force:
            print(f"{y}: exists, skipped")
            continue
        t0 = time.time()
        d = db.raw_sql(f"SELECT * FROM crsp.tfz_dly WHERE caldt BETWEEN '{y}-01-01' AND '{y}-12-31'",
                       date_cols=["caldt"])
        d.to_csv(f + ".part", index=False, compression="gzip")
        os.replace(f + ".part", f)
        log(f"{y}: {len(d)} rows, {d['kytreasno'].nunique() if 'kytreasno' in d else '?'} securities, "
            f"{d['caldt'].nunique()} days, {time.time() - t0:.0f} s")
    db.close()
    log("done")


if __name__ == "__main__":
    main()
