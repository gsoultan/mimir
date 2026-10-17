#!/bin/sh
# Runs every test on the ROCm device, then the GPU benchmarks. It runs on
# any Linux machine with an AMD GPU, ROCm 5.4 or later and Go, such as a
# container of the rocm/dev-ubuntu-24.04 image started with
#
#	docker run --device=/dev/kfd --device=/dev/dri --group-add video \
#		-v "$PWD":/mimir -w /mimir rocm/dev-ubuntu-24.04 sh ci/rocm/test.sh
#
# after installing Go there. No CI job runs it: GitHub has no AMD GPU
# runners.
set -eu
rocm-smi --showproductname || true
go run ./tools/devices | grep rocm
go test -count=1 -v -run 'Kernels|Device|HIPRTC' ./backend/rocm
MIMIR_DEVICE=rocm go test -count=1 ./...
# The same with HIP graphs, which are off by default.
MIMIR_DEVICE=rocm MIMIR_ROCM_GRAPHS=1 go test -count=1 -run 'Graphs|Batch' -v ./backend/rocm
MIMIR_DEVICE=rocm MIMIR_ROCM_GRAPHS=1 go test -count=1 ./train ./nn ./optim
echo "## benchmarks"
go test -run '^$' -bench MatMul ./backend/rocm
MIMIR_DEVICE=rocm go test -run '^$' -bench Conv2d -benchtime 20x ./tensor
for dev in cpu rocm; do
	echo "MIMIR_DEVICE=$dev"
	MIMIR_DEVICE=$dev go test -run '^$' -bench TrainStep -benchtime 3s ./train
done
