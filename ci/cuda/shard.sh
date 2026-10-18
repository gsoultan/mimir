#!/bin/sh
# Runs the sharding tests and the sharded scaling benchmark on every GPU of
# a machine with two or more NVIDIA GPUs and NCCL, such as the Modal host
# that modal_test.py starts with --gpus:
#
#	modal run ci/cuda/modal_test.py --gpus 8 --gpu-type H100 --script ci/cuda/shard.sh
#
# It ends with the 8-GPU scaling gate: the efficiency of BenchmarkSharded,
# examples per second on 8 GPUs over 8 times those on one at a fixed batch
# per GPU (steps per second on 8 over those on one, as each step on 8
# trains 8 times the examples), for the GPT-2 block and the larger
# transformer, under the strategy STRATEGY (default dp; fsdp and tp also
# run and are printed). It fails when either is below GATE (default 0.85)
# or when the host has fewer than 8 GPUs to measure it on.
#
# COUNT sets the runs of the benchmark (default 1), BENCHTIME the time of
# each (default 3s).
set -eu
count=${COUNT:-1}
benchtime=${BENCHTIME:-3s}
gate=${GATE:-0.85}
strategy=${STRATEGY:-dp}
nvidia-smi -L || true
if ! ldconfig -p | grep -q 'libnccl\.so\.2'; then
	echo "libnccl.so.2 not found: install NCCL (the nvidia/cuda images and the nvidia-nccl-cu12 wheel have it)" >&2
	exit 1
fi
go run ./tools/devices
# The collectives the partitioner inserts, on the GPUs' NCCL.
go test -count=1 -v -run 'NCCL' ./backend/cuda
# Sharded training (data, tensor and fully sharded data parallelism, and
# their products on 2×2 meshes) on the GPUs against one device, and the
# collectives each strategy issues.
MIMIR_DEVICE=gpu go test -count=1 -v -run 'Sharded' ./train
# The per-op rules on the CPU devices, for the race detector, which needs
# cgo and so a C compiler.
if [ "$(go env CGO_ENABLED)" = 1 ]; then
	go test -count=1 -race -run 'Shard|Mesh' ./tensor ./train
else
	echo "no C compiler: the race detector's run is skipped"
fi
echo "## scaling"
out=$(mktemp)
MIMIR_DEVICE=gpu go test -p 1 -run '^$' -count "$count" -bench '^BenchmarkSharded$' -benchtime "$benchtime" -timeout 60m ./train | tee "$out"
echo "## 8-GPU scaling gate ($strategy, at least $gate)"
status=0
for model in gpt2 transformer; do
	eff=$(awk -v name="BenchmarkSharded/$model/$strategy/n=8" '
		index($1, name) == 1 {
			for (i = 2; i < NF; i++) if ($(i+1) == "efficiency") { sum += $i; n++ }
		}
		END { if (n) printf "%.4f", sum / n }' "$out")
	if [ -z "$eff" ]; then
		echo "$model: no 8-GPU result (fewer than 8 GPUs?)"
		status=1
		continue
	fi
	if awk -v e="$eff" -v g="$gate" 'BEGIN { exit !(e >= g) }'; then
		echo "$model: efficiency $eff: pass"
	else
		echo "$model: efficiency $eff: FAIL"
		status=1
	fi
done
rm -f "$out"
exit $status
