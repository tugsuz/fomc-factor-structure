"""Minute prices from TAQ (millisecond) around FOMC announcements, from WRDS (9 Nov 2026).

Inputs (made by taq_targets.py, no WRDS needed):
  data_wrds/taq/targets.csv       712 days: 180 statement days, 352 control days, 180 minutes days
  data_wrds/taq/universe.csv.gz   the 500 largest common stocks of the previous close, per day

What it pulls, into data_wrds/taq/:
  crsp_tickers.csv.gz             CRSP ticker history (crsp.stksecurityinfohist) for those stocks
  bars/bars_<yyyymmdd>.csv.gz     one file per target day; one row per symbol and minute, from
                                  60 minutes before the event to 16:00:
    sym_root, sym_suffix          the TAQ symbol
    mod                           minute of the day (14:00 is 840); the row covers mod to mod + 1
    n, vol                        number of trades kept, shares traded
    p_last, p_med                 last and median trade price in that minute
The symbols are the 500 stocks of that day and a fixed list of 88 ETFs: Treasury maturities, TIPS,
credit, FX, commodities, equity size and style, sectors and international markets.

Trades kept: tr_corr = '00' (not corrected or cancelled), price > 0, and no sale condition that
marks a late, out-of-sequence, average-price or derivatively priced trade
(conditions 4 7 9 B C G H L N P R T U V W Z).

How to run (the wrds package reads ~/.pgpass, so no password is asked):
  cd code
  python3 taq_fomc_pull.py --probe
        checks tables, columns, symbols and speed on one FOMC day (30 Jan 2019). The output is
        also saved to results/taq_probe_2026-11-09.txt
  caffeinate -i python3 taq_fomc_pull.py
        the pull: statement days first, then control days, then minutes days. Days whose file
        exists are skipped, so after a dropped connection run the same command again.
        caffeinate keeps the Mac awake while it runs.
  python3 taq_fomc_pull.py --sets statement --max-days 5     a short test run
  python3 taq_fomc_pull.py --lib yearly                      read taqm_<year> instead of taqmsec
  python3 taq_fomc_pull.py --link
        the WRDS CRSP-TAQ link (wrdsapps_link_crsp_taqm.taqmclink) for the target days and stocks,
        into data_wrds/taq/taqm_link.csv.gz; used later to match TAQ symbols to CRSP permnos
The log is data_wrds/taq/pull_log.txt.
"""
import argparse
import datetime as dt
import os
import re
import sys
import time

import pandas as pd

import paths

OUT = os.path.join(paths.ROOT, "data_wrds", "taq")
BARS = os.path.join(OUT, "bars")
PROBE_OUT = os.path.join(paths.RESULTS, "taq_probe_2026-11-09.txt")
USER = os.environ.get("WRDS_USER", "")  # your WRDS user name
TEST_DAY = "2019-01-30"
BAD_CONDITIONS = "479BCGHLNPRTUVWZ"
ETFS = {
    "treasury": "SHV BIL SHY IEI IEF TLH TLT EDV GOVT VGSH VGIT VGLT TBT",
    "tips": "TIP STIP SCHP VTIP",
    "credit": "AGG BND LQD VCSH VCIT VCLT HYG JNK MBB EMB BKLN MUB",
    "fx": "UUP UDN FXE FXY FXB FXF FXA FXC",
    "commodity": "GLD IAU SLV USO UNG DBC DBA GDX",
    "equity": "SPY QQQ DIA IWM IWB IWD IWF IWN IWO MDY IJR VTI RSP",
    "sector": "XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY KBE KRE IYR VNQ XHB ITB SMH",
    "international": "EFA EEM VGK EWJ EWZ FXI EWG EWU EWC EWY",
    "volatility": "VXX VXXB",
}
ETF_LIST = [s for v in ETFS.values() for s in v.split()]
SYM_OK = re.compile(r"^[A-Z0-9]{1,6}$")


def connect():
    try:
        import wrds
    except ImportError:
        sys.exit("the wrds package is missing: python3 -m pip install wrds")
    return wrds.Connection(wrds_username=USER)


def log(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "pull_log.txt"), "a") as f:
        f.write(line + "\n")


def sql(db, q, **kw):
    """raw_sql with a rollback after an error, so that the connection stays usable."""
    try:
        return db.raw_sql(q, **kw)
    except Exception:
        try:
            db.connection.rollback()
        except Exception:
            pass
        raise


def columns(db, schema, table):
    q = ("SELECT column_name, data_type FROM information_schema.columns "
         f"WHERE table_schema = '{schema}' AND table_name = '{table}' ORDER BY ordinal_position")
    return sql(db, q)


def lib_for(day, mode):
    return "taqmsec" if mode == "taqmsec" else f"taqm_{day[:4]}"


def bars_sql(lib, day, start, end, syms):
    lst = ", ".join(f"'{s}'" for s in syms)
    return f"""
SELECT sym_root, COALESCE(sym_suffix, '') AS sym_suffix,
       CAST(EXTRACT(HOUR FROM time_m) * 60 + EXTRACT(MINUTE FROM time_m) AS integer) AS mod,
       COUNT(*) AS n, SUM(size) AS vol,
       (ARRAY_AGG(price ORDER BY time_m DESC))[1] AS p_last,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS p_med
FROM {lib}.ctm_{day.replace('-', '')}
WHERE time_m >= '{start}:00' AND time_m < '{end}:00'
  AND sym_root IN ({lst})
  AND tr_corr = '00' AND price > 0
  AND COALESCE(tr_scond, '') !~ '[{BAD_CONDITIONS}]'
GROUP BY 1, 2, 3
"""


def load_tickers(db, permnos):
    f = os.path.join(OUT, "crsp_tickers.csv.gz")
    if os.path.exists(f):
        return pd.read_csv(f, dtype={"secinfostartdt": str, "secinfoenddt": str})
    have = set(columns(db, "crsp", "stksecurityinfohist")["column_name"])
    want = [c for c in ["ticker", "tradingsymbol", "shareclass"] if c in have]
    if "ticker" not in want:
        sys.exit(f"crsp.stksecurityinfohist has no ticker column; columns: {sorted(have)}")
    plist = ", ".join(str(int(p)) for p in permnos)
    # dates are read as text (YYYY-MM-DD), so that an open end date such as 9999-12-31 stays valid
    d = sql(db, "SELECT permno, CAST(secinfostartdt AS text) AS secinfostartdt, "
                f"CAST(secinfoenddt AS text) AS secinfoenddt, {', '.join(want)} "
                f"FROM crsp.stksecurityinfohist WHERE permno IN ({plist})")
    d.to_csv(f, index=False, compression="gzip")
    log(f"crsp tickers: {len(d)} rows for {d['permno'].nunique()} permnos")
    return d


def stock_roots(day, universe, tick):
    """TAQ symbol roots of the day's 500 stocks, from the CRSP ticker valid on that day."""
    p = set(universe.loc[universe["date"] == day, "permno"])
    t = tick[tick["permno"].isin(p) & (tick["secinfostartdt"].str[:10] <= day) & (tick["secinfoenddt"].str[:10] >= day)]
    roots = set()
    for c in ["ticker", "tradingsymbol"]:
        if c in t:
            for v in t[c].dropna().astype(str):
                r = re.split(r"[.\s/]", v.strip().upper())[0]
                if SYM_OK.match(r):
                    roots.add(r)
                    if len(r) == 5:
                        # Nasdaq five-letter symbols sit in TAQ as a four-letter root and a suffix
                        # (GOOGL is GOOG + L, CMCSA is CMCS + A); the probe of 9 Nov 2026 showed this
                        roots.add(r[:4])
    return sorted(roots), len(p), t["permno"].nunique()


def probe(db, targets, universe, tick, mode):
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(str(s))

    def attempt(label, fn):
        try:
            fn()
        except Exception as e:
            say(f"  FAILED: {label}: {str(e).splitlines()[0][:200]}")

    day = TEST_DAY.replace("-", "")
    say(f"TAQ probe, {dt.datetime.now():%Y-%m-%d %H:%M}, test day {TEST_DAY}, library mode {mode}")
    say("\n1. Tables for the test day, in any library")
    attempt("tables", lambda: say(sql(db, "SELECT table_schema, table_name, table_type FROM information_schema.tables "
                                          f"WHERE right(table_name, 8) = '{day}' ORDER BY 1, 2").to_string()))
    say("\n2. Columns of ctm on the test day and on 2 Feb 2005")
    for d in [day, "20050202"]:
        attempt(f"columns {d}", lambda d=d: say(f"{lib_for(d, mode)}.ctm_{d}:\n" + columns(
            db, lib_for(d, mode), f"ctm_{d}").to_string()))
    say("\n3. Coverage: which libraries have a ctm table on these FOMC days")
    for d in ["20030916", "20040128", "20050202", "20080318", "20251210"]:
        attempt(f"coverage {d}", lambda d=d: say(f"  {d}: " + ", ".join(sql(db,
            "SELECT table_schema FROM information_schema.tables "
            f"WHERE table_name = 'ctm_{d}'")["table_schema"].tolist())))
    lib = lib_for(day, mode)
    say(f"\n4. SPY trades 14:00 to 14:05 on the test day, by correction code and sale condition")
    attempt("tr_corr", lambda: say(sql(db, f"SELECT tr_corr, COUNT(*) AS n FROM {lib}.ctm_{day} WHERE sym_root = 'SPY' "
                                           "AND time_m >= '14:00:00' AND time_m < '14:05:00' GROUP BY 1 ORDER BY 2 DESC").to_string()))
    attempt("tr_scond", lambda: say(sql(db, f"SELECT tr_scond, COUNT(*) AS n FROM {lib}.ctm_{day} WHERE sym_root = 'SPY' "
                                            "AND time_m >= '14:00:00' AND time_m < '14:05:00' GROUP BY 1 ORDER BY 2 DESC LIMIT 15").to_string()))
    say("\n5. CRSP ticker columns used for the stock symbols")
    say("  " + ", ".join(c for c in tick.columns))
    say("\n6. CRSP to TAQ link tables (for a later check of the symbol match)")
    attempt("link", lambda: say(sql(db, "SELECT table_schema, table_name, string_agg(column_name, ', ' ORDER BY ordinal_position) AS cols "
                                        "FROM information_schema.columns WHERE table_schema IN "
                                        "('wrdsapps_link_crsp_taqm', 'wrdsapps_link_crsp_taq') GROUP BY 1, 2").to_string()))
    say("\n7. Minute bars for SPY, AAPL and TLT, 13:58 to 14:03")
    attempt("small bars", lambda: say(sql(db, bars_sql(lib, TEST_DAY, "13:58", "14:03", ["SPY", "AAPL", "TLT"])).sort_values(
        ["sym_root", "mod"]).to_string()))
    say("\n8. Speed: the full query for the test day")
    r = targets[targets["date"] == TEST_DAY].iloc[0]
    roots, n_perm, n_tick = stock_roots(TEST_DAY, universe, tick)
    say(f"  stocks {n_perm}, with a CRSP ticker {n_tick}, symbol roots {len(roots)}; ETFs {len(ETF_LIST)}; "
        f"window {r['start']} to {r['end']}")

    def full():
        t0 = time.time()
        b = sql(db, bars_sql(lib, TEST_DAY, r["start"], r["end"], sorted(set(roots) | set(ETF_LIST))))
        s = time.time() - t0
        got = set(b["sym_root"])
        say(f"  {len(b)} rows in {s:.1f} s; stock roots with trades {len(got & set(roots))} of {len(roots)}; "
            f"ETFs with trades {len(got & set(ETF_LIST))} of {len(ETF_LIST)}")
        say(f"  stock roots without trades: {sorted(set(roots) - got)[:40]}")
        say(f"  ETFs without trades: {sorted(set(ETF_LIST) - got)}")
        say(f"  suffixes seen: {b['sym_suffix'].value_counts().head(10).to_dict()}")
        n = len(targets)
        say(f"  rough total for all {n} days: {s * n / 3600:.1f} hours (recent days are the heaviest, "
            f"so this is an upper end), about {len(b) * n / 1e6:.0f} million rows")
    attempt("full", full)
    os.makedirs(os.path.dirname(PROBE_OUT), exist_ok=True)
    open(PROBE_OUT, "w").write("\n".join(lines) + "\n")
    print(f"\nSaved to {PROBE_OUT}")


def pull_link(db, targets, universe):
    f = os.path.join(OUT, "taqm_link.csv.gz")
    dates = ", ".join(f"'{d}'" for d in sorted(set(targets["date"])))
    plist = ", ".join(str(int(p)) for p in sorted(universe["permno"].unique()))
    t0 = time.time()
    d = sql(db, "SELECT CAST(date AS text) AS date, sym_root, COALESCE(sym_suffix, '') AS sym_suffix, "
                "permno, match_lvl FROM wrdsapps_link_crsp_taqm.taqmclink "
                f"WHERE date IN ({dates}) AND permno IN ({plist})")
    d.to_csv(f, index=False, compression="gzip")
    log(f"taqm link: {len(d)} rows, {d['date'].nunique()} days, {d['permno'].nunique()} permnos, "
        f"{time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sets", default="statement,control,minutes")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--lib", choices=["taqmsec", "yearly"], default="taqmsec")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--start", default="2005-01-01",
                    help="first day; the ASU subscription cannot read taqm_2003 and taqm_2004 (9 Nov 2026)")
    a = ap.parse_args()
    targets = pd.read_csv(os.path.join(OUT, "targets.csv"), dtype=str)
    targets = targets[targets["date"] >= a.start].reset_index(drop=True)
    universe = pd.read_csv(os.path.join(OUT, "universe.csv.gz"), dtype={"date": str})
    db = connect()
    tick = load_tickers(db, universe["permno"].unique())
    if a.probe:
        probe(db, targets, universe, tick, a.lib)
        db.close()
        return
    if a.link:
        pull_link(db, targets, universe)
        db.close()
        return
    os.makedirs(BARS, exist_ok=True)
    sets = a.sets.split(",")
    todo = targets[targets["set"].isin(sets)]
    todo = todo.iloc[sorted(range(len(todo)), key=lambda i: sets.index(todo["set"].iloc[i]))]
    done = 0
    errors = 0
    secs = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        day = r.date
        f = os.path.join(BARS, f"bars_{day.replace('-', '')}.csv.gz")
        if os.path.exists(f) and not a.force:
            continue
        roots, n_perm, n_tick = stock_roots(day, universe, tick)
        syms = sorted(set(roots) | set(ETF_LIST))
        t0 = time.time()
        try:
            b = sql(db, bars_sql(lib_for(day, a.lib), day, r.start, r.end, syms))
        except Exception as e:
            msg = str(e).splitlines()[0][:160]
            if "does not exist" in msg or "UndefinedTable" in msg or "permission denied" in msg:
                log(f"{i}/{len(todo)} {day} {r.set}: skipped, no TAQ table or no access ({msg})")
                continue
            errors += 1
            log(f"{i}/{len(todo)} {day} {r.set}: ERROR {msg}")
            if errors >= 5:
                sys.exit("five errors; stopping. Run the same command again to resume.")
            try:
                db.close()
            except Exception:
                pass
            time.sleep(10)
            db = connect()
            continue
        errors = 0
        for c in ["p_last", "p_med"]:
            b[c] = b[c].astype(float).round(4)
        b.sort_values(["sym_root", "sym_suffix", "mod"]).to_csv(f + ".part", index=False, compression="gzip")
        os.replace(f + ".part", f)
        secs.append(time.time() - t0)
        done += 1
        left = sum(1 for x in todo.itertuples(index=False)
                   if not os.path.exists(os.path.join(BARS, f"bars_{x.date.replace('-', '')}.csv.gz")))
        log(f"{i}/{len(todo)} {day} {r.set}: {len(b)} rows, {b['sym_root'].nunique()} symbols "
            f"({n_tick} of {n_perm} stocks with a ticker), {secs[-1]:.0f} s; "
            f"about {sum(secs) / len(secs) * left / 3600:.1f} h left")
        if a.max_days and done >= a.max_days:
            break
    db.close()
    log("done")


if __name__ == "__main__":
    main()
