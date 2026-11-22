"""Scheduled macro-release days: data_public/macro_release_days.csv.

Releases and sources:
  empsit   Employment Situation     BLS archive page bls.gov/bls/news-release/empsit.htm   FRED release id 50
  cpi      Consumer Price Index     BLS archive page bls.gov/bls/news-release/cpi.htm      FRED release id 10
  ppi      Producer Price Index     BLS archive page bls.gov/bls/news-release/ppi.htm      FRED release id 46
  gdp      Gross Domestic Product   FRED release id 53 only
  pi       Personal Income and Outlays   FRED release id 54 only

The three BLS archive pages list every past release since 1994 with the release date in the link
URL (for example news.release/archives/empsit_10022026.htm is 2 October 2026); this script reads
those dates with a regular expression, nothing else. The FRED files are the JSON answers of
fred/release/dates (one per release id, full history, include_release_dates_with_no_data=true).
Both sets are produced by code/download_macro_dates.sh, run from a terminal; the FRED
part needs an API key in the environment variable FRED_API_KEY (free, fredaccount.stlouisfed.org/apikeys).

Inputs, whichever exist in data_public/macro/:
  bls_empsit.htm, bls_cpi.htm, bls_ppi.htm           parsed here (deterministic)
  fred_release_dates_<rid>.json, rid in 10 46 50 53 54
  bls_release_dates_provisional.csv                   the fallback for the three BLS releases: the same
                                                      link dates, read off the pages on 28 Oct 2026 without
                                                      the pages being saved; replaced as soon as the pages exist

Output: data_public/macro_release_days.csv with columns date, release, source, and a report on
stdout (and data_public/macro/macro_release_dates_report.txt): which sources were used, the
cross-check of BLS pages against FRED for the three BLS releases, and the months without a release.

Usage: python macro_release_dates.py
"""
import glob
import json
import os
import re

import pandas as pd

import paths

MACRO_DIR = os.path.join(paths.DATA, "macro")
OUT_CSV = os.path.join(paths.DATA, "macro_release_days.csv")
BLS = {"empsit": "bls_empsit.htm", "cpi": "bls_cpi.htm", "ppi": "bls_ppi.htm"}
FRED = {"cpi": 10, "ppi": 46, "empsit": 50, "gdp": 53, "pi": 54}
START = pd.Timestamp("1994-01-01")


def parse_bls_page(path, release):
    html = open(path, encoding="utf-8", errors="replace").read()
    found = re.findall(rf"{release}_(\d{{2}})(\d{{2}})(\d{{4}})\.(?:htm|pdf|txt)", html)
    dates = sorted({pd.Timestamp(int(y), int(m), int(d)) for m, d, y in found})
    return pd.DatetimeIndex(dates)


def parse_fred_json(path):
    j = json.load(open(path))
    dates = sorted({pd.Timestamp(x["date"]) for x in j.get("release_dates", [])})
    return pd.DatetimeIndex(dates)


def months_without(dates, start, end):
    ym = pd.DatetimeIndex(dates).to_period("M")
    allm = pd.period_range(start, end, freq="M")
    return [str(m) for m in allm if m not in set(ym)]


def main():
    os.makedirs(MACRO_DIR, exist_ok=True)
    report = []
    rows = []
    fred = {}
    for rel, rid in FRED.items():
        p = os.path.join(MACRO_DIR, f"fred_release_dates_{rid}.json")
        if os.path.exists(p):
            fred[rel] = parse_fred_json(p)
            report.append(f"FRED {rel} (release id {rid}): {len(fred[rel])} dates, {fred[rel].min().date()} to {fred[rel].max().date()}")
    prov = None
    pprov = os.path.join(MACRO_DIR, "bls_release_dates_provisional.csv")
    if os.path.exists(pprov):
        prov = pd.read_csv(pprov, parse_dates=["date"])
    for rel, fname in BLS.items():
        p = os.path.join(MACRO_DIR, fname)
        if os.path.exists(p):
            d = parse_bls_page(p, rel)
            src = f"BLS archive page {fname}"
        elif prov is not None:
            d = pd.DatetimeIndex(prov.loc[prov.release == rel, "date"]).sort_values()
            src = "BLS archive link dates, provisional list of 28 Oct 2026"
        elif rel in fred:
            d = fred[rel]
            src = f"FRED release id {FRED[rel]}"
        else:
            report.append(f"{rel}: no source available")
            continue
        d = d[d >= START]
        report.append(f"{rel}: {len(d)} dates from {src}; months without a release: {months_without(d, '1994-02', d.max().to_period('M'))}")
        if rel in fred and not src.startswith("FRED"):
            f = fred[rel][fred[rel] >= START]
            only_src = sorted(set(d) - set(f))
            only_fred = sorted(set(f) - set(d))
            report.append(f"   cross-check against FRED id {FRED[rel]}: in the BLS list only {[x.date().isoformat() for x in only_src]}; in FRED only {[x.date().isoformat() for x in only_fred]}")
        for x in d:
            rows.append({"date": x.date().isoformat(), "release": rel, "source": src})
    for rel in ("gdp", "pi"):
        if rel in fred:
            d = fred[rel][fred[rel] >= START]
            for x in d:
                rows.append({"date": x.date().isoformat(), "release": rel, "source": f"FRED release id {FRED[rel]}"})
        else:
            report.append(f"{rel}: not available (needs the FRED pull: export FRED_API_KEY and run download_macro_dates.sh)")
    out = pd.DataFrame(rows).sort_values(["date", "release"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    report.append(f"written {OUT_CSV}: {len(out)} rows, {out.date.nunique()} distinct days, releases {sorted(out.release.unique())}")
    text = "\n".join(report)
    open(os.path.join(MACRO_DIR, "macro_release_dates_report.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
