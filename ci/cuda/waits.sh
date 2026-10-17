#!/bin/sh
# Times how long the GPUs take in each phase of a data-parallel step
# (MIMIR_DP_WAITS=1, see BenchmarkDataParallel), for the gpt2 and cnn x8
# steps on 1, 2 and every GPU, with CUDA graphs on and off:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/waits.sh
set -eu
nvidia-smi -L || true
gpus=$(nvidia-smi -L | grep -c '^GPU')
for graphs in 1 0; do
	echo "## MIMIR_CUDA_GRAPHS=$graphs"
	MIMIR_CUDA_GRAPHS=$graphs MIMIR_DEVICE=gpu MIMIR_DP_WAITS=1 go test -run '^$' -count 1 -benchtime 3s \
		-bench "^BenchmarkDataParallel\$/^(gpt2|cnn)\$/^x8\$/^(1|2|$gpus)\$" ./train
done
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv || true
