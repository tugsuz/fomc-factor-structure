"""Target days and stock universe for the TAQ pull around FOMC announcements (9 Nov 2026).

What it makes, in data_wrds/taq/:
  targets.csv       one row per day to pull: the date, the set (statement, control, minutes), the
                    event time, the start and end of the pull window, and for control days the
                    FOMC day they are paired with
  universe.csv.gz   for each target day, the 500 largest common stocks (CRSP permno) by market cap
                    at the close of the previous trading day

The three sets:
  statement  FOMC statement days from USMPD (sheet Statements), 10 Sep 2003 to 31 Dec 2025, with a
             release time between 09:45 and 15:30, so that the market is open before and after it.
             Millisecond TAQ starts on 10 Sep 2003.
  control    for every scheduled statement day, the same weekday one week before and one week
             after, if it is a trading day and not a statement or minutes day. It uses the clock
             time of the paired statement. These days are the placebo: same clock time, no FOMC.
  minutes    FOMC minutes release days from USMPD (sheet Minutes), same period and time filter.
The pull window runs from 60 minutes before the event time to 16:00.

Runs on the project files only (no WRDS):  python3 taq_targets.py
"""
import os

import numpy as np
import pandas as pd

import paths

OUT = os.path.join(paths.ROOT, "data_wrds", "taq")
USMPD = os.path.join(paths.FED, "USMPD.xlsx")
START, END = pd.Timestamp("2003-09-10"), pd.Timestamp("2025-12-31")
YEARS = range(2003, 2026)
TOP = 500


def crsp_file(y):
    return os.path.join(paths.ROOT, "data_wrds", f"crsp_daily_{y}.csv.gz")


def usmpd(sheet):
    d = pd.read_excel(USMPD, sheet)
    d = d[(d["Date"] >= START) & (d["Date"] <= END)].copy()
    d["date"] = d["Date"].dt.normalize()
    d["time"] = d["date_time"].dt.strftime("%H:%M")
    return d


def minus(hhmm, minutes):
    t = pd.Timestamp("2000-01-01 " + hhmm) - pd.Timedelta(minutes=minutes)
    return t.strftime("%H:%M")


def main():
    os.makedirs(OUT, exist_ok=True)
    # trading days (from the CRSP files we already have)
    tdays = pd.DatetimeIndex(sorted(set(np.concatenate(
        [pd.to_datetime(pd.read_csv(crsp_file(y), usecols=["date"])["date"].unique()) for y in YEARS]))))
    tset = set(tdays)

    st, pc, mi = usmpd("Statements"), usmpd("Press Conferences"), usmpd("Minutes")
    all_statement_days = set(st["date"])
    minutes_days = set(mi["date"])
    pc_time = dict(zip(pc["date"], pc["time"]))

    rows = []
    s = st[st["time"].between("09:45", "15:30") & st["date"].isin(tset)]
    for _, r in s.iterrows():
        rows.append(dict(date=r["date"], set="statement", event_time=r["time"],
                         unscheduled=int(r["Unscheduled"]), sep=int(r["SEP"]),
                         press_conf=int(r["date"] in pc_time), press_conf_time=pc_time.get(r["date"], ""),
                         pair=""))
    taken = set(s["date"])
    for _, r in s[s["Unscheduled"] == 0].iterrows():
        for k in (-7, 7):
            c = r["date"] + pd.Timedelta(days=k)
            if c in tset and c not in all_statement_days and c not in minutes_days and c not in taken:
                rows.append(dict(date=c, set="control", event_time=r["time"], unscheduled=0, sep=0,
                                 press_conf=0, press_conf_time="", pair=r["date"].strftime("%Y-%m-%d")))
                taken.add(c)
    m = mi[mi["time"].between("09:45", "15:30") & mi["date"].isin(tset)]
    for _, r in m.iterrows():
        if r["date"] not in taken:
            rows.append(dict(date=r["date"], set="minutes", event_time=r["time"], unscheduled=0, sep=0,
                             press_conf=0, press_conf_time="", pair=""))
            taken.add(r["date"])
    t = pd.DataFrame(rows)
    t["start"] = t["event_time"].map(lambda x: minus(x, 60))
    t["end"] = "16:00"
    t["prev_day"] = [tdays[tdays.searchsorted(d) - 1] for d in t["date"]]
    order = {"statement": 0, "control": 1, "minutes": 2}
    t = t.sort_values(["set", "date"], key=lambda c: c.map(order) if c.name == "set" else c).reset_index(drop=True)

    # the 500 largest common stocks at the previous close
    need = set(t["prev_day"])
    caps = []
    for y in range(2003, 2026):
        d = pd.read_csv(crsp_file(y), usecols=["permno", "date", "cap"], parse_dates=["date"])
        d = d[d["date"].isin(need) & d["cap"].notna() & (d["cap"] > 0)]
        d = d.sort_values(["date", "cap"], ascending=[True, False])
        d["rank"] = d.groupby("date").cumcount() + 1
        caps.append(d[d["rank"] <= TOP])
    caps = pd.concat(caps).rename(columns={"date": "prev_day", "cap": "cap_prev"})
    u = t[["date", "prev_day"]].merge(caps, on="prev_day")[["date", "permno", "rank", "cap_prev"]]

    t_out = t.copy()
    for c in ["date", "prev_day"]:
        t_out[c] = t_out[c].dt.strftime("%Y-%m-%d")
    t_out.to_csv(os.path.join(OUT, "targets.csv"), index=False)
    u["date"] = u["date"].dt.strftime("%Y-%m-%d")
    u.to_csv(os.path.join(OUT, "universe.csv.gz"), index=False, compression="gzip")

    print("target days by set:", t["set"].value_counts().to_dict())
    print("statement days by event time:", t[t["set"] == "statement"]["event_time"].value_counts().to_dict())
    print("statement days with a press conference:", int(t["press_conf"].sum()))
    print("universe rows:", len(u), " permnos:", u["permno"].nunique(),
          " days with fewer than 500 stocks:", int((u.groupby("date").size() < TOP).sum()))
    print(t.groupby([t["date"].dt.year, "set"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
