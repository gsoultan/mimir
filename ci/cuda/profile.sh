#!/bin/sh
# Profiles the host side of data-parallel training on every GPU of the
# machine: a CPU profile of BenchmarkDataParallel's gpt2 x8 step on one
# GPU and on all of them, printed as the functions that take the most
# time, so that what the host does for each replica can be compared:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/profile.sh
set -eu
nvidia-smi -L || true
gpus=$(nvidia-smi -L | grep -c '^GPU')
out=$(mktemp -d)
for n in 1 "$gpus"; do
	echo "## gpt2/x8 on $n GPU(s)"
	MIMIR_DEVICE=gpu go test -run '^$' -count 1 -benchtime 5s \
		-bench "^BenchmarkDataParallel\$/^gpt2\$/^x8\$/^$n\$" \
		-cpuprofile "$out/cpu$n.pprof" -o "$out/train.test" ./train
	echo "### by cumulative time"
	go tool pprof -top -cum -nodecount 60 "$out/train.test" "$out/cpu$n.pprof" 2>/dev/null | head -75
	echo "### by own time"
	go tool pprof -top -nodecount 40 "$out/train.test" "$out/cpu$n.pprof" 2>/dev/null | head -55
done
