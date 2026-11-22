"""Compustat firm traits for the stocks of the TAQ universe, from WRDS (17 Nov 2026).

Why: the panel models of Ahn, Lee and Schmidt use firm traits as instruments. So far we have traits from
CRSP only (sector, size, betas, volatility). Compustat adds leverage, book-to-market, cash and debt maturity.

What it pulls, into data_wrds/compustat/:
  ccm_link.csv.gz       the gvkey-permno link records for the permnos in data_wrds/taq/universe.csv.gz
  funda_<year>.csv.gz   annual fundamentals (comp.funda) of the linked firms, one file for each fiscal year

How to run (the wrds package reads ~/.pgpass, so no password is asked):
  cd code
  python3 compustat_pull.py --probe     lists the tables and columns it would use and pulls nothing
  python3 compustat_pull.py             pulls; it takes a few minutes
The script can be started again at any time. A file that is already on disk is skipped.
It asks WRDS for the column names first, so a missing column is reported and not requested.
Run --probe first to check the tables and columns.
"""
import os
import sys

import pandas as pd

import paths

OUT = os.path.join(paths.ROOT, "data_wrds", "compustat")
TAQ = os.path.join(paths.ROOT, "data_wrds", "taq")
USER = os.environ.get("WRDS_USER", "")  # your WRDS user name
YEARS = range(2003, 2026)
WANT = ["gvkey", "datadate", "fyear", "at", "lt", "dltt", "dlc", "dd1", "che", "ceq", "seq", "pstk", "txditc",
        "sale", "ib", "oibdp", "xint", "capx", "ppent", "csho", "prcc_f", "sich"]
LINKS = [("crsp", "ccmxpf_lnkhist"), ("crsp_a_ccm", "ccmxpf_lnkhist")]


def columns(db, schema, table):
    q = f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema}' AND table_name = '{table}'"
    return db.raw_sql(q)["column_name"].tolist()


def save(d, path):
    tmp = path + ".part"
    d.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def main(probe):
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: python3 -m pip install wrds")
    os.makedirs(OUT, exist_ok=True)
    u = pd.read_csv(os.path.join(TAQ, "universe.csv.gz"), usecols=["permno"])
    permnos = sorted(int(p) for p in u["permno"].unique())
    db = wrds.Connection(wrds_username=USER)
    link = next(((s, t) for s, t in LINKS if columns(db, s, t)), None)
    fcols = columns(db, "comp", "funda")
    have = [c for c in WANT if c in fcols]
    print("link table:", link, "| comp.funda columns found:", len(have), "of", len(WANT), "| missing:", [c for c in WANT if c not in fcols])
    if link is None or not have:
        db.close()
        sys.exit("a table is missing or cannot be read")
    if probe:
        n = db.raw_sql(f"SELECT COUNT(*) AS n FROM {link[0]}.{link[1]} WHERE lpermno IN ({', '.join(map(str, permnos[:50]))})")
        print("probe: link rows for the first 50 permnos:", int(n["n"][0]), "| permnos in the universe:", len(permnos))
        db.close()
        return
    lf = os.path.join(OUT, "ccm_link.csv.gz")
    if not os.path.exists(lf):
        l = db.raw_sql(f"SELECT gvkey, lpermno AS permno, linktype, linkprim, CAST(linkdt AS text) AS linkdt, "
                       f"CAST(linkenddt AS text) AS linkenddt FROM {link[0]}.{link[1]} "
                       f"WHERE lpermno IN ({', '.join(map(str, permnos))}) AND linktype IN ('LU', 'LC')")
        save(l, lf)
    l = pd.read_csv(lf, dtype={"gvkey": str})
    gv = ", ".join("'" + g + "'" for g in sorted(l["gvkey"].unique()))
    print("linked firms:", l["gvkey"].nunique())
    sel = ", ".join("CAST(datadate AS text) AS datadate" if c == "datadate" else c for c in have)
    for y in YEARS:
        f = os.path.join(OUT, f"funda_{y}.csv.gz")
        if os.path.exists(f):
            print(y, "already on disk")
            continue
        d = db.raw_sql(f"SELECT {sel} FROM comp.funda WHERE fyear = {y} AND gvkey IN ({gv}) "
                       f"AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'")
        save(d, f)
        print(y, "rows:", len(d), flush=True)
    db.close()


if __name__ == "__main__":
    main("--probe" in sys.argv)
