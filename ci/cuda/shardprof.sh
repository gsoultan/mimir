#!/bin/sh
# Profiles the host's work in the sharded training step of
# BenchmarkSharded on the GPUs of the machine, such as the Modal host that
# modal_test.py starts with --gpus:
#
#	modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/shardprof.sh
#
# For the GPT-2 block and the transformer it times the step on one GPU
# without a mesh (n=1), on a mesh of one GPU (mesh=1), whose difference is
# the partitioner's cost, and under data parallelism on every GPU, each
# with a CPU profile of which it prints the functions of mimir that take
# the most time, with what they call. BENCHTIME sets the time of each
# (default 3s).
set -eu
benchtime=${BENCHTIME:-3s}
n=$(nvidia-smi -L | wc -l)
nvidia-smi -L
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
go test -c -o "$tmp/train.test" ./train
cd train
for model in gpt2 transformer; do
	for sub in n=1 mesh=1 "dp/n=$n"; do
		# Each level of the name is matched alone: ^dp$ leaves out fsdp.
		pattern="^BenchmarkSharded\$/^$model\$/^$(echo "$sub" | sed 's|/|$/^|')\$"
		echo "## $model/$sub"
		MIMIR_DEVICE=gpu "$tmp/train.test" -test.run '^$' -test.bench "$pattern" \
			-test.benchtime "$benchtime" -test.cpuprofile "$tmp/cpu.out" | grep '^Benchmark'
		go tool pprof -top -cum -nodecount 45 -focus 'mimir/(tensor|internal|optim|nn)' \
			-ignore '^runtime\.|^testing\.' "$tmp/train.test" "$tmp/cpu.out" 2>/dev/null |
			sed -n '/flat%/,$p'
	done
done
