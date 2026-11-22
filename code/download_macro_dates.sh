#!/bin/bash
# Downloads the sources of the scheduled macro-release calendar into term_paper/data_public/macro/.
# Run it from a terminal with internet access:
#   bash code/download_macro_dates.sh
# Part 1 (no key): the three BLS news-release archive pages (Employment Situation, CPI, PPI).
# Part 2 (needs a FRED API key, free at https://fredaccount.stlouisfed.org/apikeys):
#   export FRED_API_KEY=your32characterkey   before running; it pulls the release-date history
#   of release ids 10 (CPI), 46 (PPI), 50 (Employment Situation), 53 (GDP), 54 (Personal Income and Outlays).
# Then: python3 code/macro_release_dates.py   (writes data_public/macro_release_days.csv)
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data_public/macro"
mkdir -p "$D"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

for r in empsit cpi ppi; do
  curl -fsSL -A "$UA" -o "$D/bls_$r.htm" "https://www.bls.gov/bls/news-release/$r.htm" && echo "ok      BLS $r" || echo "FAILED  BLS $r"
  sleep 1
done

if [ -n "${FRED_API_KEY:-}" ]; then
  for rid in 10 46 50 53 54; do
    curl -fsSL -o "$D/fred_release_dates_$rid.json" \
      "https://api.stlouisfed.org/fred/release/dates?release_id=$rid&include_release_dates_with_no_data=true&file_type=json&api_key=$FRED_API_KEY" \
      && echo "ok      FRED release $rid" || echo "FAILED  FRED release $rid"
    sleep 1
  done
else
  echo "FRED_API_KEY not set: skipped the FRED release-date pull (GDP and personal income dates need it)"
fi
ls -la "$D"
