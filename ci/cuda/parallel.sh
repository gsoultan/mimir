#!/bin/sh
# Runs the data-parallel tests and the scaling benchmark on every GPU of a
# machine with two or more NVIDIA GPUs and NCCL, such as the Modal host
# that modal_test.py starts with --gpus:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/parallel.sh
#
# COUNT sets the runs of the benchmark (default 1).
set -eu
count=${COUNT:-1}
nvidia-smi -L || true
if ! ldconfig -p | grep -q 'libnccl\.so\.2'; then
	echo "libnccl.so.2 not found: install NCCL (the nvidia/cuda images and the nvidia-nccl-cu12 wheel have it)" >&2
	exit 1
fi
go run ./tools/devices
# NCCL's collectives between the devices of one process.
go test -count=1 -v -run 'NCCL' ./backend/cuda
# Collectives, replicas and data-parallel training on the GPUs, against
# the same training on one device.
MIMIR_DEVICE=gpu go test -count=1 -v -run 'Group|ReduceGrads|Replicate|Parallel' ./tensor ./nn ./train
echo "## scaling"
MIMIR_DEVICE=gpu go test -p 1 -run '^$' -count "$count" -bench '^BenchmarkDataParallel$' -benchtime 3s ./train
