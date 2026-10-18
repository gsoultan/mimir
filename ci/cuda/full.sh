#!/bin/sh
# Runs every CUDA check in one job: the tests on the GPU (test.sh), the
# benchmarks with CUDA graphs off and on (perf.sh), and, given enough
# GPUs, the data-parallel tests and scaling (parallel.sh, 2+ GPUs) and the
# sharding gate (shard.sh, 8 GPUs). A part that fails does not stop the
# others; the summary at the end lists each, and the exit code is 1 if
# any failed.
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/full.sh
#
# COUNT is passed on to the benchmarks.
set -u
cd "$(dirname "$0")/../.."
gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
echo "## $gpus GPU(s)"

summary=""
failed=0
part() {
	name=$1 script=$2
	echo
	echo "################ $name ($script) ################"
	start=$(date +%s)
	if sh "$script"; then
		result=PASS
	else
		result=FAIL
		failed=1
	fi
	summary="$summary
$result  $name  $(($(date +%s) - start))s"
}
skip() {
	summary="$summary
SKIP  $1  ($2)"
}

part "tests" ci/cuda/test.sh
part "benchmarks, graphs off and on" ci/cuda/perf.sh
if [ "$gpus" -ge 2 ]; then
	part "data parallel on $gpus GPUs" ci/cuda/parallel.sh
else
	skip "data parallel" "needs 2+ GPUs, have $gpus"
fi
if [ "$gpus" -ge 8 ]; then
	part "sharding gate on $gpus GPUs" ci/cuda/shard.sh
else
	skip "sharding gate" "needs 8 GPUs, have $gpus"
fi

echo
echo "################ summary ################$summary"
exit $failed
