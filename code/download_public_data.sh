#!/bin/bash
# Downloads the public data of the project into data_public/.
# Run it from a terminal with internet access:
#   bash code/download_public_data.sh
set -u
D="$(cd "$(dirname "$0")/.." && pwd)/data_public"
mkdir -p "$D/french" "$D/fed"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
FR="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Fama-French daily factors and daily portfolios
for f in F-F_Research_Data_Factors_daily F-F_Research_Data_5_Factors_2x3_daily F-F_Momentum_Factor_daily \
         100_Portfolios_10x10_Daily 100_Portfolios_ME_OP_10x10_daily 100_Portfolios_ME_INV_10x10_daily \
         25_Portfolios_5x5_Daily 25_Portfolios_ME_Prior_12_2_Daily 10_Portfolios_Prior_12_2_Daily \
         49_Industry_Portfolios_daily; do
  curl -fsSL -A "$UA" -o "$D/french/${f}_CSV.zip" "$FR/${f}_CSV.zip" && echo "ok      $f" || echo "FAILED  $f"
done

# GSW zero-coupon yields, Bauer-Swanson surprises (end 2023), SF Fed USMPD (to 2026), FOMC calendars
curl -fsSL -A "$UA" -o "$D/fed/feds200628.csv" \
  "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv" && echo "ok      GSW" || echo "FAILED  GSW"
curl -fsSL -A "$UA" -o "$D/fed/monetary-policy-surprises-data.xlsx" \
  "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx" && echo "ok      Bauer-Swanson" || echo "FAILED  Bauer-Swanson"
curl -fsSL -A "$UA" -o "$D/fed/USMPD.xlsx" \
  "https://www.frbsf.org/wp-content/uploads/USMPD.xlsx" && echo "ok      USMPD" || echo "FAILED  USMPD"
curl -fsSL -A "$UA" -o "$D/fed/usmpd-monetary-policy-surprises.zip" \
  "https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises.zip" && echo "ok      USMPD surprises" || echo "FAILED  USMPD surprises"
curl -fsSL -A "$UA" -o "$D/fed/fomccalendars.htm" \
  "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm" && echo "ok      FOMC calendar 2021-2027" || echo "FAILED  FOMC calendar"
for y in $(seq 1994 2020); do
  curl -fsSL -A "$UA" -o "$D/fed/fomchistorical$y.htm" \
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical$y.htm" && echo "ok      FOMC $y" || echo "FAILED  FOMC $y"
  sleep 1
done

echo
du -sh "$D/french" "$D/fed"
