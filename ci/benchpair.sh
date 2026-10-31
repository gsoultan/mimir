#!/usr/bin/env bash
# Benchmarks the base and the pull request alternately, so that the two
# runs of each package are seconds apart and the drift of a shared runner,
# whose speed changes over the hour the benchmarks take, affects both.
# Running the whole base after the whole pull request made unrelated
# benchmarks look 20% slower, run after run.
#
#   ci/benchpair.sh BASE_DIR HEAD_DIR COUNT base.txt head.txt
#
# Each package with benchmarks in HEAD_DIR is compiled from both trees,
# then the two binaries run their benchmarks in turn COUNT times, from the
# package's directory. A package the base lacks runs for the pull request
# only. The outputs are in the format go test prints, for benchstat.
set -euo pipefail
base=$1 head=$2 count=$3 baseout=$4 headout=$5
bin=$(mktemp -d)
trap 'rm -rf "$bin"' EXIT
: > "$baseout"
: > "$headout"
pkgs=$(cd "$head" && grep -rl --include='*_test.go' '^func Benchmark' . | xargs -n1 dirname | sort -u)
for pkg in $pkgs; do
  rm -f "$bin/base.test" "$bin/head.test"
  (cd "$head" && go test -c -o "$bin/head.test" "$pkg")
  if [ -d "$base/$pkg" ]; then
    (cd "$base" && go test -c -o "$bin/base.test" "$pkg") || true
  fi
  for _ in $(seq "$count"); do
    if [ -x "$bin/base.test" ]; then
      (cd "$base/$pkg" && "$bin/base.test" -test.run='^$' -test.bench=. -test.count=1 >> "$baseout")
    fi
    (cd "$head/$pkg" && "$bin/head.test" -test.run='^$' -test.bench=. -test.count=1 | tee -a "$headout")
  done
done
