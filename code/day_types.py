"""Day types for the FOMC factor project.

For a given index of trading days (the panel's dates) this module builds the boolean columns

  fomc         scheduled FOMC announcement days (fomc_meetings.csv, kind == scheduled; the Fed's
               15 Sep 2003 entry is already tagged other_meeting by fomc_dates.py)
  pre_fomc     the trading day before each scheduled announcement day (pre-FOMC drift)
  post_fomc    the trading day after each scheduled announcement day (information only; kept in the pool)
  fed_event    unscheduled meetings, conference calls, notation votes, cancelled and other meetings
               listed on the Fed's own pages
  usmpd_unsch  the unscheduled statement days of the USMPD Statements sheet (18 through Sep 2026);
               ten of them are intermeeting actions that are not on the Fed's meeting pages
  macro        scheduled macro-release days from data_public/macro_release_days.csv (Employment
               Situation, CPI, PPI; GDP and personal income when the FRED pull has been run)
  pool         none of the above: the no-news pool

and a `type` column with the priority fomc > usmpd_unsch > fed_event > pre_fomc > macro > pool.
A day keeps every flag it earns; the priority only decides `type`.

The USMPD check: every scheduled announcement day in the sample must be a scheduled statement
in USMPD, and every scheduled USMPD statement up to the panel's last day must be an announcement
day (fomc_dates.py and USMPD agree exactly through 29 July 2026; checked 26 Oct 2026).

Usage as a module: from day_types import day_types, pool_id
Usage as a script: python day_types.py   (prints the counts for the portfolio panel's dates)
"""
import hashlib
import os

import numpy as np
import pandas as pd

import paths

MACRO_CSV = os.path.join(paths.DATA, "macro_release_days.csv")
USMPD_XLSX = os.path.join(paths.FED, "USMPD.xlsx")
FED_EVENT_KINDS = ["unscheduled", "conference_call", "notation_vote", "cancelled", "other_meeting"]


def load_fomc(meetings_csv=paths.MEETINGS):
    m = pd.read_csv(meetings_csv, parse_dates=["date"])
    sched = pd.DatetimeIndex(m.loc[m.kind == "scheduled", "date"]).sort_values()
    events = pd.DatetimeIndex(m.loc[m.kind.isin(FED_EVENT_KINDS), "date"]).sort_values()
    return sched, events


def load_usmpd_statements(xlsx=USMPD_XLSX):
    """Scheduled and unscheduled statement days of the USMPD Statements sheet."""
    u = pd.read_excel(xlsx, sheet_name="Statements", usecols=["Date", "Unscheduled"])
    d = pd.to_datetime(u["Date"]).dt.normalize()
    unsch = pd.DatetimeIndex(d[u["Unscheduled"] == 1]).sort_values().unique()
    sched = pd.DatetimeIndex(d[u["Unscheduled"] == 0]).sort_values().unique()
    return sched, unsch


def load_macro(csv=MACRO_CSV):
    """Macro-release days as a DatetimeIndex, with the per-release table; empty if the file is missing."""
    if not os.path.exists(csv):
        return pd.DatetimeIndex([]), None
    m = pd.read_csv(csv, parse_dates=["date"])
    return pd.DatetimeIndex(m["date"]).sort_values().unique(), m


def day_types(dates, meetings_csv=paths.MEETINGS, usmpd_xlsx=USMPD_XLSX, macro_csv=MACRO_CSV, check_usmpd=True):
    dates = pd.DatetimeIndex(dates)
    sched, events = load_fomc(meetings_csv)
    df = pd.DataFrame(index=dates)
    df["fomc"] = dates.isin(sched)
    pos = np.flatnonzero(df["fomc"].values)
    pre = np.zeros(len(dates), bool)
    pre[pos[pos > 0] - 1] = True
    post = np.zeros(len(dates), bool)
    post[pos[pos < len(dates) - 1] + 1] = True
    df["pre_fomc"] = pre
    df["post_fomc"] = post
    df["fed_event"] = dates.isin(events)
    if usmpd_xlsx and os.path.exists(usmpd_xlsx):
        u_sched, u_unsch = load_usmpd_statements(usmpd_xlsx)
        df["usmpd_unsch"] = dates.isin(u_unsch)
        if check_usmpd:
            last = min(dates[-1], u_sched[-1])
            a = set(sched[(sched >= dates[0]) & (sched <= last)])
            b = set(u_sched[(u_sched >= dates[0]) & (u_sched <= last)])
            if a != b:
                raise ValueError(f"FOMC calendar and USMPD disagree: only in calendar {sorted(a - b)}, only in USMPD {sorted(b - a)}")
    else:
        df["usmpd_unsch"] = False
    macro_days, _ = load_macro(macro_csv)
    df["macro"] = dates.isin(macro_days)
    df["pool"] = ~(df.fomc | df.pre_fomc | df.fed_event | df.usmpd_unsch | df.macro)
    typ = np.full(len(dates), "pool", dtype=object)
    for name in ["macro", "pre_fomc", "fed_event", "usmpd_unsch", "fomc"]:      # increasing priority
        typ[df[name].values] = name
    df["type"] = typ
    return df


def summary(df):
    lines = [f"{len(df)} trading days, {df.index[0].date()} to {df.index[-1].date()}"]
    for name in ["fomc", "pre_fomc", "post_fomc", "fed_event", "usmpd_unsch", "macro", "pool"]:
        lines.append(f"  {name:12s} {int(df[name].sum()):5d}")
    lines.append("  type counts: " + ", ".join(f"{k} {v}" for k, v in df["type"].value_counts().items()))
    both = df[df.fomc & df.macro]
    if len(both):
        lines.append("  announcement days that are also macro-release days (kept as fomc): " +
                     ", ".join(d.strftime("%Y-%m-%d") for d in both.index))
    return "\n".join(lines)


def pool_id(df):
    """Short hash of the pool definition, stored with the saved placebo draws."""
    key = ",".join(d.strftime("%Y%m%d") for d in df.index[df.pool.values])
    return hashlib.sha1(key.encode()).hexdigest()[:10]


if __name__ == "__main__":
    from load_data import portfolio_panel
    X, _ = portfolio_panel(paths.FRENCH, start="1994-01-01")
    dt = day_types(X.index)
    print(summary(dt))
    print("pool id:", pool_id(dt))
