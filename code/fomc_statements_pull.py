"""Download the FOMC statements from the Federal Reserve site and save them as text (19 Nov 2026).

Part of the Entity Neutering extension. No key is needed.
The User-Agent is a generic string, or the environment variable FED_USER_AGENT if you set one.

Steps
  1. list   The links to the statements are read from the FOMC calendar pages that are already in
            data_public/fed/ (fomchistoricalYYYY.htm and fomccalendars.htm). No network is used.
  2. pull   Each statement page is downloaded once and kept as raw HTML in
            data_public/fomc_statements/html/. A page that is already there is not downloaded again.
  3. parse  The text of the statement is cut out of the HTML. The list of voters is stored apart,
            because the names give the year away. A sentence about a dissent is kept apart too.

Output: data_public/fomc_statements/urls.csv, html/<date>.html, statements.csv with the columns
  date, url, status, n_words, text, vote, dissent.
  status: ok, check (fewer than 60 or more than 1,500 words: look at the page), no page, error.

Usage, from inside code/:
  python3 fomc_statements_pull.py --selftest      test of the parser, no network
  python3 fomc_statements_pull.py --list          step 1 only, no network
  python3 fomc_statements_pull.py --dry-run       says what it would download
  python3 fomc_statements_pull.py --limit 5       a first try with five pages
  python3 fomc_statements_pull.py                 all pages (about one second per page)
  python3 fomc_statements_pull.py --parse-only    cut the text again from the saved HTML
"""
import argparse
import csv
import glob
import html as htmllib
import os
import re
import sys
import time
import urllib.request

import paths

BASE = "https://www.federalreserve.gov"
OUT = os.path.join(paths.DATA, "fomc_statements")
HTML = os.path.join(OUT, "html")
UA = os.environ.get("FED_USER_AGENT", "Mozilla/5.0 (research script, statements only, one request per second)")

START = re.compile(r"(?i)^(for (immediate )?release|for release at|release date|press release|share$)")
END = re.compile(r"(?i)^(voting for|last update|implementation note|for media inquiries|return to top|related information|"
                 r"home \||\d{4} (monetary|general|press|banking)|back to top|accessibility|to comment on this site|"
                 r"board of governors of the federal reserve system$)")
BODY = re.compile(r"(?i)\b(committee|federal reserve|board of governors|federal funds|discount rate)\b")


def find_links():
    """(date, url, page) for every statement link in the saved calendar pages."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(paths.FED, "fomc*.htm"))):
        page = open(path, encoding="utf-8", errors="ignore").read()
        for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.S | re.I):
            label = re.sub(r"<[^>]+>|\s+", " ", label).strip()
            m = re.search(r"(\d{8})", href)
            if not m or not re.search(r"/press", href, re.I) or href.lower().endswith(".pdf"):
                continue
            old = label.lower() == "statement"                                   # pages up to 2020
            new = label.upper() == "HTML" and re.search(r"monetary\d{8}a\.htm$", href)   # current calendar page
            if old or new:
                d = m.group(1)
                date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                rows.setdefault(date, (date, href if href.startswith("http") else BASE + href, os.path.basename(path)))
    return [rows[k] for k in sorted(rows)]


def to_lines(page):
    page = re.sub(r"(?is)<(script|style|noscript|head|nav|footer|form)\b.*?</\1>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|td|tr|li|h\d|table|ul|blockquote)>", "\n", page)
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", page)).replace("\xa0", " ")
    return [re.sub(r"\s+", " ", l).strip() for l in text.split("\n") if l.strip()]


def extract(page):
    """Returns (text, vote, dissent).
    The body ends at the list of voters, or at the first closing line of the page if there is no such list.
    It starts after the last release line before that point ("For immediate release", "For release at", "Share")."""
    lines = to_lines(page)
    iv = next((i for i, l in enumerate(lines) if re.match(r"(?i)^voting (for|against)", l)), None)
    if iv is not None:
        end = iv
    else:
        s0 = next((i for i, l in enumerate(lines) if START.match(l)), 0)
        end = next((i for i in range(s0 + 1, len(lines))
                    if END.match(lines[i]) and any(len(x.split()) >= 12 for x in lines[s0:i])), len(lines))
    starts = [i for i in range(end) if START.match(lines[i])]
    body = [l for l in lines[(starts[-1] + 1 if starts else 0):end] if len(l.split()) >= 4]
    while body and (len(body[0].split()) < 12 or (not starts and not BODY.search(body[0]))):
        body.pop(0)
    vote = []
    for l in lines[iv:] if iv is not None else []:
        if END.match(l) and not re.match(r"(?i)^voting", l):
            break
        vote.append(l)
    vote = " ".join(vote)
    m = re.search(r"(?i)voting against.*", vote)
    return "\n".join(body), vote, (m.group(0) if m else "")


def parse_all(links):
    rows = []
    for date, url, _ in links:
        path = os.path.join(HTML, date + ".html")
        if not os.path.exists(path):
            rows.append([date, url, "no page", 0, "", "", ""])
            continue
        text, vote, dissent = extract(open(path, encoding="utf-8", errors="ignore").read())
        n = len(text.split())
        rows.append([date, url, "ok" if 60 <= n <= 1500 else "check", n, text, vote, dissent])
    with open(os.path.join(OUT, "statements.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "url", "status", "n_words", "text", "vote", "dissent"])
        w.writerows(rows)
    count = {}
    for r in rows:
        count[r[2]] = count.get(r[2], 0) + 1
    print("statements.csv:", count, "| words:", sum(r[3] for r in rows))
    for r in rows:
        if r[2] == "check":
            print("  check", r[0], r[3], "words", r[1])


def selftest():
    new = ("<html><head><title>x</title></head><body><nav>Menu Home</nav><div><h3>Press Release</h3><p>March 22, 2023</p>"
           "<h3>Federal Reserve issues FOMC statement</h3><p>For release at 2:00 p.m. EDT</p><div><p>Recent indicators point to "
           "modest growth in spending and production. Job gains have picked up in recent months and are running at a robust pace.</p>"
           "<p>The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. "
           "The Committee decided to raise the target range for the federal funds rate.</p>"
           "<p>Voting for the monetary policy action were Jerome H. Powell, Chair; John C. Williams, Vice Chair. Voting against "
           "this action was A. Member, who preferred no change.</p><p>Implementation Note issued March 22, 2023</p></div>"
           "<p>Last Update: March 22, 2023</p></div></body></html>")
    old = ("<html><body><table><tr><td><font><b>Release Date: January 28, 2004</b></font><p>For immediate release</p>"
           "<p>The Federal Open Market Committee decided today to keep its target for the federal funds rate at 1 percent.</p>"
           "<p>The Committee continues to believe that an accommodative stance of monetary policy, coupled with robust underlying "
           "growth in productivity, is providing important ongoing support to economic activity.</p>"
           "<p>Voting for the FOMC monetary policy action were: Alan Greenspan, Chairman; Timothy F. Geithner.</p>"
           "<p>2004 Monetary policy</p><p>Home | News and events</p></td></tr></table></body></html>")
    for name, page, first, n_par in [("new", new, "Recent indicators", 2), ("old", old, "The Federal Open Market Committee decided", 2)]:
        text, vote, dissent = extract(page)
        assert text.startswith(first), (name, text[:80])
        assert len(text.split("\n")) == n_par, (name, text)
        assert "Voting for" in vote and "Implementation" not in text and "Home |" not in text, (name, vote)
    assert extract(new)[2].startswith("Voting against")
    assert extract(old)[2] == ""
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    os.makedirs(HTML, exist_ok=True)
    links = find_links()
    with open(os.path.join(OUT, "urls.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "url", "calendar_page"])
        w.writerows(links)
    by_year = {}
    for d, _, _ in links:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
    print(f"{len(links)} statement links, {links[0][0]} to {links[-1][0]}. By year: {by_year}")
    if a.list:
        return
    todo = [l for l in links if not os.path.exists(os.path.join(HTML, l[0] + ".html"))]
    if a.limit:
        todo = todo[:a.limit]
    if a.dry_run:
        print(f"would download {len(todo)} pages (about {len(todo)} seconds). Already saved: {len(links) - len(todo)}.")
        return
    if not a.parse_only:
        for i, (date, url, _) in enumerate(todo):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                page = urllib.request.urlopen(req, timeout=30).read()
                with open(os.path.join(HTML, date + ".html"), "wb") as fh:
                    fh.write(page)
                print(f"{i + 1}/{len(todo)} {date} {len(page)} bytes")
            except Exception as e:          # the page stays missing, so the next run tries it again
                print(f"{i + 1}/{len(todo)} {date} error: {e}")
            time.sleep(1.0)
    parse_all(links)


if __name__ == "__main__":
    main()
