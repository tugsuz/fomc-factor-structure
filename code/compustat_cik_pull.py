"""CIK numbers for the Compustat firms of the TAQ universe, from WRDS (17 Nov 2026).

Notation: nothing is counted here. k is not used (k is the number of regressors in the panel lecture notes).

Why: SEC EDGAR lists the filings of a firm by its CIK number. edgar_item7a_pull.py needs the CIK of each firm.

What it pulls, into data_wrds/compustat/:
  company_cik.csv.gz    gvkey, cik and conm (the company name) from comp.company, for the firms in ccm_link.csv.gz

How to run (the wrds package reads ~/.pgpass, so no password is asked):
  cd code
  python3 compustat_cik_pull.py --probe     checks the table and its columns and pulls nothing
  python3 compustat_cik_pull.py             pulls; it takes a few seconds
The script can be started again at any time. If the file is on disk, it does nothing.
Run --probe first to check the tables and columns.
"""
import os
import sys

import pandas as pd

from compustat_pull import OUT, USER, columns, save


def main(probe):
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: python3 -m pip install wrds")
    f = os.path.join(OUT, "company_cik.csv.gz")
    if os.path.exists(f) and not probe:
        print("already on disk:", f)
        return
    link = pd.read_csv(os.path.join(OUT, "ccm_link.csv.gz"), dtype={"gvkey": str})
    gv = sorted(link["gvkey"].unique())
    quoted = lambda g: ", ".join("'" + x + "'" for x in g)
    db = wrds.Connection(wrds_username=USER)
    cols = columns(db, "comp", "company")
    want = [c for c in ["gvkey", "cik", "conm"] if c in cols]
    print("comp.company columns found:", want)
    if "gvkey" not in want or "cik" not in want:
        db.close()
        sys.exit("comp.company has no gvkey or no cik column")
    if probe:
        n = db.raw_sql(f"SELECT COUNT(*) AS n FROM comp.company WHERE gvkey IN ({quoted(gv[:50])}) AND cik IS NOT NULL")
        print("probe: firms with a CIK among the first 50:", int(n["n"][0]), "| firms to pull:", len(gv))
        db.close()
        return
    d = db.raw_sql(f"SELECT {', '.join(want)} FROM comp.company WHERE gvkey IN ({quoted(gv)})")
    db.close()
    save(d, f)
    print("firms:", len(d), "| with a CIK:", int(d["cik"].notna().sum()))


if __name__ == "__main__":
    main("--probe" in sys.argv)
