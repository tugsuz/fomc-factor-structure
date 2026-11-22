"""Build the list of FOMC meetings from the Federal Reserve's calendar pages.

Inputs (saved by download_public_data.sh, in data_public/):
  fed/fomchistorical1994.htm ... fomchistorical2020.htm   (one page per year)
  fed/fomccalendars.htm                                   (2021 onward)
Output: data_public/fomc_meetings.csv with one row per meeting or conference call:
  date        the announcement day (last day of the meeting)
  start       first day of the meeting
  kind        scheduled / unscheduled / cancelled / conference_call /
              notation_vote / other_meeting
  sep         True if the meeting has a Summary of Economic Projections (2021 on only)
  label       the heading as printed on the Fed page

Scheduled meetings are the ones with kind == "scheduled". The script checks
that every complete year has 8 of them (2020 has 7 because March was cancelled).

"""
import csv
import datetime as dt
import glob
import os
import re
import sys

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
for name, i in list(MONTHS.items()):
    MONTHS[name[:3]] = i
MONTHS["sept"] = 9


def month_number(text):
    return MONTHS[text.strip().lower().rstrip(".")]


def make_dates(year, month_text, day_text):
    """Return (start, end) dates for headings such as 'January 31-February 1',
    'Feb 31-1' (the Fed's own shorthand for Jan 31 to Feb 1) or 'Jan/Feb 31-1'."""
    day_text = day_text.strip()
    months = [m for m in re.split(r"[/\s]+", month_text.strip()) if m]
    m_end = month_number(months[-1])
    m_start = month_number(months[0])
    parts = re.findall(r"([A-Za-z]+\s+)?(\d{1,2})", day_text)
    d_start = int(parts[0][1])
    d_end = int(parts[-1][1])
    if parts[-1][0]:                       # 'January 31-February 1'
        m_end = month_number(parts[-1][0])
    if d_end < d_start and m_start == m_end:
        m_start = m_end - 1 if m_end > 1 else 12   # 'Feb 31-1' means Jan 31 to Feb 1
    y_start = year - 1 if (m_start == 12 and m_end == 1) else year
    return dt.date(y_start, m_start, d_start), dt.date(year, m_end, d_end)


HIST = re.compile(
    r"([A-Z][a-z]{2,8})\.?\s+(\d{1,2}(?:\s*-\s*(?:[A-Z][a-z]{2,8}\.?\s+)?\d{1,2})?)\s*"
    r"(\((?:unscheduled|cancelled)\)\s*)?(Meeting|Conference Call)\s*-\s*(\d{4})")


def parse_historical(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    text = re.sub(r"<[^>]+>", " ", text)
    rows, seen = [], set()
    for mon, days, flag, what, year in HIST.findall(text):
        if mon.lower().rstrip(".") not in MONTHS:
            continue
        year = int(year)
        start, end = make_dates(year, mon, days)
        if what == "Conference Call":
            kind = "conference_call"
        elif flag:
            kind = "unscheduled" if "unscheduled" in flag else "cancelled"
        else:
            kind = "scheduled"
        label = re.sub(r"\s+", " ", f"{mon} {days} {flag}{what} - {year}")
        key = (end, kind)
        if key not in seen:
            seen.add(key)
            rows.append({"date": end, "start": start, "kind": kind, "sep": "",
                         "label": label})
    return rows


def parse_calendar(path):
    html = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    blocks = re.split(r">\s*(20\d\d) FOMC Meetings\s*<", html)
    # blocks = [before, year1, body1, year2, body2, ...]
    for year, body in zip(blocks[1::2], blocks[2::2]):
        year = int(year)
        pairs = re.findall(
            r'fomc-meeting__month[^>]*>\s*(?:<strong>)?([^<]+?)(?:</strong>)?\s*</div>.*?'
            r'fomc-meeting__date[^>]*>\s*([^<]+?)\s*</div>', body, flags=re.S)
        for mon, days in pairs:
            raw = days
            sep = "*" in days
            low = days.lower()
            if "notation vote" in low:
                kind = "notation_vote"
            elif "unscheduled" in low:
                kind = "unscheduled"
            elif "cancel" in low:
                kind = "cancelled"
            else:
                kind = "scheduled"
            days = re.sub(r"\(.*?\)|\*", "", days)
            start, end = make_dates(year, mon, days)
            rows.append({"date": end, "start": start, "kind": kind, "sep": sep,
                         "label": f"{mon.strip()} {raw.strip()} - {year}"})
    return rows


# The Fed's 2003 page lists a September 15 meeting the day before the scheduled
# September 16 meeting. The policy statement came on September 16 (the
# Bauer-Swanson file agrees), so September 15 is not an announcement day.
NOT_ANNOUNCEMENTS = {dt.date(2003, 9, 15)}


def main(fed_dir, out_csv):
    rows = []
    for path in sorted(glob.glob(os.path.join(fed_dir, "fomchistorical*.htm"))):
        rows += parse_historical(path)
    rows += parse_calendar(os.path.join(fed_dir, "fomccalendars.htm"))
    rows = sorted({(r["date"], r["kind"]): r for r in rows}.values(), key=lambda r: r["date"])
    for r in rows:
        if r["date"] in NOT_ANNOUNCEMENTS and r["kind"] == "scheduled":
            r["kind"] = "other_meeting"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "start", "kind", "sep", "label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # checks
    years = sorted({r["date"].year for r in rows})
    print("year  scheduled  unscheduled  conf_calls  other")
    bad = []
    for y in years:
        k = [r["kind"] for r in rows if r["date"].year == y]
        s = k.count("scheduled")
        print(f"{y}  {s:9d}  {k.count('unscheduled'):11d}  {k.count('conference_call'):10d}  "
              f"{len(k) - s - k.count('unscheduled') - k.count('conference_call'):5d}")
        expected = 7 if y == 2020 else 8
        if s != expected:
            bad.append((y, s))
    print("years with an unexpected number of scheduled meetings:", bad or "none")
    return rows


if __name__ == "__main__":
    import paths
    fed_dir = sys.argv[1] if len(sys.argv) > 1 else paths.FED
    out_csv = sys.argv[2] if len(sys.argv) > 2 else paths.MEETINGS
    main(fed_dir, out_csv)
