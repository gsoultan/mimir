#!/bin/sh
# Tells whether GPUs slow each other down outside mimir's data-parallel
# code: times the single-GPU gpt2 and cnn x8 steps alone on GPU 0, then
# in one process per GPU, all at once, then the data-parallel step on
# every GPU, sampling each GPU's power, clocks and PCIe traffic with
# nvidia-smi dmon throughout. It first checks that training steps over
# batches uploaded every step launch as CUDA graphs:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/contend.sh
set -eu
nvidia-smi -L || true
gpus=$(nvidia-smi -L | grep -c '^GPU')
go test -count=1 -v -run TestTrainingStepsRunAsCUDAGraphs ./backend/cuda
tmp=$(mktemp -d)
bin=$tmp/train.test
go test -c -o "$bin" ./train
bench() {
	MIMIR_DEVICE=gpu "$bin" -test.run '^$' -test.count 1 -test.benchtime 3s -test.bench "$1"
}
one='^BenchmarkDataParallel$/^(gpt2|cnn)$/^x8$/^1$'
nvidia-smi dmon -s pucet -o T >"$tmp/dmon.txt" 2>&1 &
dmon=$!
echo "## alone on GPU 0"
CUDA_VISIBLE_DEVICES=0 bench "$one" | grep Benchmark
echo "## one process per GPU, at once"
i=0
pids=
while [ "$i" -lt "$gpus" ]; do
	CUDA_VISIBLE_DEVICES=$i bench "$one" >"$tmp/alone$i.txt" 2>&1 &
	pids="$pids $!"
	i=$((i + 1))
done
for p in $pids; do
	wait "$p" || true
done
i=0
while [ "$i" -lt "$gpus" ]; do
	echo "# GPU $i"
	grep Benchmark "$tmp/alone$i.txt" || cat "$tmp/alone$i.txt"
	i=$((i + 1))
done
echo "## data parallel on $gpus GPUs"
bench "^BenchmarkDataParallel\$/^(gpt2|cnn)\$/^x8\$/^$gpus\$" | grep Benchmark
kill "$dmon" 2>/dev/null || true
echo "## nvidia-smi dmon"
cat "$tmp/dmon.txt"
