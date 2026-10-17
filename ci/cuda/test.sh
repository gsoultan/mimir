#!/bin/sh
# Runs every test on the CUDA device, then the GPU benchmarks. The CI job
# runs it on a Modal T4 through modal_test.py; it runs the same way on any
# Linux machine with an NVIDIA GPU and Go.
set -eu
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
go run ./tools/devices | grep cuda
go test -count=1 -v -run 'Kernels|Device' ./backend/cuda
MIMIR_DEVICE=gpu go test -count=1 ./...
echo "## benchmarks"
go test -run '^$' -bench MatMul ./backend/cuda
MIMIR_DEVICE=gpu go test -run '^$' -bench Conv2d -benchtime 20x ./tensor
for dev in cpu gpu; do
	echo "MIMIR_DEVICE=$dev"
	MIMIR_DEVICE=$dev go test -run '^$' -bench TrainStep -benchtime 3s ./train
done
