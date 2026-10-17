#!/bin/sh
# Profiles data-parallel training on every GPU of the machine. First the
# gpt2 x8 step of BenchmarkDataParallel on one GPU and on all of them
# with MIMIR_DP_WAITS=1, which waits for the GPUs after each phase and
# reports how long they took in each; then CPU profiles of the same
# steps without waiting, printed as the functions that take the most
# time:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/profile.sh
set -eu
nvidia-smi -L || true
gpus=$(nvidia-smi -L | grep -c '^GPU')
out=$(mktemp -d)
echo "## the GPUs' time in each phase"
MIMIR_DEVICE=gpu MIMIR_DP_WAITS=1 go test -run '^$' -count 1 -benchtime 3s \
	-bench "^BenchmarkDataParallel\$/^(gpt2|cnn)\$/^x8\$/^(1|$gpus)\$" ./train
for n in 1 "$gpus"; do
	echo "## gpt2/x8 on $n GPU(s)"
	MIMIR_DEVICE=gpu go test -run '^$' -count 1 -benchtime 5s \
		-bench "^BenchmarkDataParallel\$/^gpt2\$/^x8\$/^$n\$" \
		-cpuprofile "$out/cpu$n.pprof" -o "$out/train.test" ./train
	echo "### by cumulative time"
	go tool pprof -top -cum -nodecount 60 "$out/train.test" "$out/cpu$n.pprof" 2>/dev/null | head -75
	echo "### what Parallel.Backward calls"
	go tool pprof -peek 'Parallel\).Backward$|Tensor\).Backward$|RunBatch$' "$out/train.test" "$out/cpu$n.pprof" 2>/dev/null | head -80
done
