#!/bin/sh
# Times how long the GPUs take in each phase of a data-parallel step
# (MIMIR_DP_WAITS=1, see BenchmarkDataParallel), for the gpt2 and cnn x8
# steps on 1, 2 and every GPU, after the tests of copies and collectives
# between GPUs, with the optimizer stepping every replica and the model
# alone (MIMIR_DP_STEP), and each without the waits too:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/waits.sh
set -eu
nvidia-smi -L || true
gpus=$(nvidia-smi -L | grep -c '^GPU')
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH go test -count=1 -v -run 'CopyPeer|NCCL' ./backend/cuda
MIMIR_DEVICE=gpu go test -count=1 -run 'Group|ReduceGrads|Replicate|Parallel|BetweenDevices' ./tensor ./nn ./train
for step in local model; do
	echo "## MIMIR_DP_STEP=$step"
	MIMIR_DP_STEP=$step MIMIR_DEVICE=gpu MIMIR_DP_WAITS=1 go test -run '^$' -count 1 -benchtime 3s \
		-bench "^BenchmarkDataParallel\$/^(gpt2|cnn)\$/^x8\$/^(1|2|$gpus)\$" ./train
	echo "## MIMIR_DP_STEP=$step, without waits"
	MIMIR_DP_STEP=$step MIMIR_DEVICE=gpu go test -run '^$' -count 1 -benchtime 3s \
		-bench "^BenchmarkDataParallel\$/^(gpt2|cnn)\$/^x8\$/^$gpus\$" ./train
done
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv || true
