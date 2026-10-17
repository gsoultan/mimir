#!/bin/sh
# Runs the ROCm benchmarks that track the GPU's speed, each with HIP
# graphs off and on (MIMIR_ROCM_GRAPHS), one package at a time so that no
# two benchmarks share the device. It runs on any Linux machine with an
# AMD GPU, ROCm and Go, as test.sh does.
#
# COUNT sets the runs of each benchmark (default 3), for benchstat.
set -eu
count=${COUNT:-3}
rocm-smi --showproductname || true
go run ./tools/devices | grep rocm

bench() {
	# bench graphs package pattern [go test flags...]
	graphs=$1 pkg=$2 pattern=$3
	shift 3
	echo "## MIMIR_ROCM_GRAPHS=$graphs $pkg $pattern"
	MIMIR_DEVICE=rocm MIMIR_ROCM_GRAPHS=$graphs go test -p 1 -run '^$' -count "$count" -bench "$pattern" "$@" "$pkg"
}

# The products, on hipBLAS and on the generated kernel, which no graph
# runs.
bench 0 ./backend/rocm '^BenchmarkMatMul$'

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
