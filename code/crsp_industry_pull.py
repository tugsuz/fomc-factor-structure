"""Industry codes for the stocks of the TAQ universe, from WRDS (10 Nov 2026).

Why: the extra factor of the FOMC window loads on banks and insurers against utilities and gold
miners. To form sector groups (for industry portfolios, or for the multilevel method of Ahn and
Zhang 2023) the project needs industry codes. The CRSP files we pulled have none.

What it pulls, into data_wrds/taq/crsp_industry.csv.gz:
  permno, the start and end date of each record, and the industry columns that exist in
  crsp.stksecurityinfohist (siccd, naics, icbindustry), for the permnos in universe.csv.gz.

How to run (the wrds package reads ~/.pgpass, so no password is asked):
  cd code
  python3 crsp_industry_pull.py
It takes a few seconds. The script lists the columns of the table first, so a missing column is
reported and not requested.
"""
import os
import sys

import pandas as pd

import paths

OUT = os.path.join(paths.ROOT, "data_wrds", "taq")
USER = os.environ.get("WRDS_USER", "")  # your WRDS user name
WANT = ["siccd", "naics", "icbindustry", "securitynm", "issuernm"]


def main():
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: python3 -m pip install wrds")
    u = pd.read_csv(os.path.join(OUT, "universe.csv.gz"), usecols=["permno"])
    permnos = sorted(u["permno"].unique())
    db = wrds.Connection(wrds_username=USER)
    cols = db.raw_sql("SELECT column_name FROM information_schema.columns "
                      "WHERE table_schema = 'crsp' AND table_name = 'stksecurityinfohist'")["column_name"].tolist()
    have = [c for c in WANT if c in cols]
    print("columns found:", have, "| not in the table:", [c for c in WANT if c not in cols])
    if not have:
        print("all columns of the table:", cols)
        sys.exit("no industry column found")
    plist = ", ".join(str(int(p)) for p in permnos)
    d = db.raw_sql("SELECT permno, CAST(secinfostartdt AS text) AS secinfostartdt, CAST(secinfoenddt AS text) AS secinfoenddt, "
                   f"{', '.join(have)} FROM crsp.stksecurityinfohist WHERE permno IN ({plist})")
    db.close()
    f = os.path.join(OUT, "crsp_industry.csv.gz")
    d.to_csv(f, index=False, compression="gzip")
    print(f"{len(d)} rows for {d['permno'].nunique()} of {len(permnos)} permnos, saved to {f}")
    if "siccd" in d:
        last = d.sort_values("secinfoenddt").drop_duplicates("permno", keep="last")
        print("stocks by first SIC digit (latest record):", last["siccd"].astype(str).str[:1].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
