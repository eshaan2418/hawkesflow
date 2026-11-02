#!/usr/bin/env bash
# Extract displayed, aggressor-signed executions for a few symbols during regular hours from the
# tickstore CSV export of the NASDAQ ITCH 2019-01-30 sample (ts = ns since midnight ET).
# Output: one CSV per symbol with ts_ns,side(+1 buy aggressor / -1 sell aggressor),size,price.
set -euo pipefail
SRC=${1:-$HOME/quantfactory/_data/tickstore/csv/trades.csv}
OUT=${2:-$HOME/quantfactory/_data/hawkesflow}
SYMS=${SYMS:-AAPL,MSFT,AMZN,NVDA,TSLA,QQQ}
mkdir -p "$OUT"
awk -F, -v syms="$SYMS" -v out="$OUT" '
  BEGIN { n = split(syms, s, ","); for (i = 1; i <= n; i++) want[s[i]] = 1 }
  NR > 1 && ($1 in want) {
    f = $5 + 0
    if (f >= 4) next                        # crosses (4) and non-displayed (8) excluded
    if ($2 < 34200000000000 || $2 >= 57600000000000) next   # 09:30-16:00
    side = (f == 1) ? 1 : -1
    print $2 "," side "," $4 "," $3 > (out "/" $1 ".csv")
  }' "$SRC"
wc -l "$OUT"/*.csv
