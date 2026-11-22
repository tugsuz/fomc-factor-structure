#!/bin/bash
# Downloads Ken French's MONTHLY portfolio and factor files into term_paper/data_public/french_monthly/.
# Run it from a terminal with internet access:
#   bash code/download_french_monthly.sh
# Then run again:  python3 monthly_counts.py   (it uses these files when they are there;
# without them it builds monthly returns by compounding the daily files).
# The file names were checked against two R packages that list them (tidyfinance, JFE) on 10 Nov 2026.
# A wrong name prints FAILED and the rest still downloads.
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data_public/french_monthly"
mkdir -p "$D"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
FR="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

for f in F-F_Research_Data_Factors F-F_Research_Data_5_Factors_2x3 F-F_Momentum_Factor \
         25_Portfolios_5x5 100_Portfolios_10x10 \
         25_Portfolios_ME_OP_5x5 25_Portfolios_ME_INV_5x5 \
         25_Portfolios_BEME_OP_5x5 25_Portfolios_BEME_INV_5x5 25_Portfolios_OP_INV_5x5 \
         25_Portfolios_ME_Prior_12_2 10_Portfolios_Prior_12_2 \
         100_Portfolios_ME_OP_10x10 100_Portfolios_ME_INV_10x10 \
         49_Industry_Portfolios 30_Industry_Portfolios 17_Industry_Portfolios; do
  curl -fsSL -A "$UA" -o "$D/${f}_CSV.zip" "$FR/${f}_CSV.zip" && echo "ok      $f" || echo "FAILED  $f"
  sleep 1
done
echo
ls -l "$D"
