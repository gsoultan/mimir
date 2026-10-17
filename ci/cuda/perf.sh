#!/bin/sh
# Runs the CUDA benchmarks that track the GPU's speed, each with CUDA
# graphs off and on (MIMIR_CUDA_GRAPHS), one package at a time so that no
# two benchmarks share the device. It runs on any Linux machine with an
# NVIDIA GPU and Go, such as the T4 that modal_test.py starts:
#
#	modal run ci/cuda/modal_test.py --script ci/cuda/perf.sh
#
# COUNT sets the runs of each benchmark (default 3), for benchstat.
set -eu
count=${COUNT:-3}
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || true
go run ./tools/devices | grep cuda

bench() {
	# bench graphs package pattern [go test flags...]
	graphs=$1 pkg=$2 pattern=$3
	shift 3
	echo "## MIMIR_CUDA_GRAPHS=$graphs $pkg $pattern"
	MIMIR_DEVICE=gpu MIMIR_CUDA_GRAPHS=$graphs go test -p 1 -run '^$' -count "$count" -bench "$pattern" "$@" "$pkg"
}

# The products, on cuBLAS and on the generated kernel, which no graph
# runs.
bench 0 ./backend/cuda '^BenchmarkMatMul$'

for graphs in 0 1; do
	# Whole training steps, whose launches graphs replay.
	bench $graphs ./train '^BenchmarkTrainStep$' -benchtime 3s
	# The host's time per step outside the device, and, under /gpu, the
	# time it waits for the device.
	bench $graphs ./tensor '^BenchmarkStepOverhead$' -benchtime 2s
	bench $graphs ./tensor '^BenchmarkConv2d$' -benchtime 20x
	bench $graphs ./tensor '^BenchmarkAttention(Forward)?$' -benchtime 10x
	bench $graphs ./nn '^BenchmarkDecoderBlockStep$' -benchtime 10x
	bench $graphs ./nn '^BenchmarkDecode$' -benchtime 5x
	bench $graphs ./nn '^BenchmarkLSTM$' -benchtime 10x
done

# The CPU's training step, for the speedup.
echo "## MIMIR_DEVICE=cpu ./train TrainStep"
MIMIR_DEVICE=cpu go test -p 1 -run '^$' -count "$count" -bench '^BenchmarkTrainStep$' -benchtime 3s ./train
