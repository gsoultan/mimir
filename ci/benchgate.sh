#!/usr/bin/env bash
# Fails when a benchmark got slower: benchstat's CSV comparison of the base
# branch and the pull request has a time per op more than LIMIT percent
# higher (20 by default) with p below 0.05. Shared runners vary by several
# percent between runs, so smaller changes and ones benchstat cannot tell
# from noise pass.
#
#   ci/benchgate.sh base.txt head.txt
set -euo pipefail
limit="${LIMIT:-20}"
benchstat -format csv "$1" "$2" 2>/dev/null | awk -F, -v limit="$limit" '
  $2 == "sec/op" { timing = 1; next }
  $2 != "" && $3 == "CI" { timing = 0 }       # another unit block
  timing && $1 != "geomean" && $6 ~ /^\+[0-9.]+%$/ {
    delta = substr($6, 2, length($6) - 2) + 0
    p = $7; sub(/^p=/, "", p); sub(/ .*/, "", p)
    if (delta > limit && p + 0 < 0.05) {
      printf "slower: %s by %s (%s)\n", $1, $6, $7
      bad = 1
    }
  }
  END { exit bad }'
