#!/bin/sh
# Runs M6's speed gates: mimir's transformer and recurrent benchmarks
# against PyTorch doing the same work on the same GPU. It times in
# PyTorch, with tools/bench/attention.py, causal attention at GPT-2 small
# shapes, a GPT-2 small block's training step, a decoding step at Llama
# 3.2 1B shapes and a 2-layer LSTM, then the matching benchmarks of
# packages tensor and nn with CUDA graphs on, and fails when mimir takes
# more than GATE times PyTorch's median time on any of them:
#
#	modal run ci/cuda/modal_test.py --script ci/cuda/m6.sh
#
# GATE defaults to 1.25. COUNT sets the runs of each mimir benchmark
# (default 3), whose median is compared. PyTorch is installed once into
# the cache volume that modal_test.py mounts at /cache.
set -eu
gate=${GATE:-1.25}
count=${COUNT:-3}
torch=${TORCH_VERSION:-2.8.0}
out=$(mktemp -d)
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || true

site=${CACHE:-/cache}/torch-$torch
if [ ! -d "$site/torch" ]; then
	python3 -m pip install --quiet --target "$site" "torch==$torch" \
		--index-url https://download.pytorch.org/whl/cu126
fi

echo "## PyTorch $torch"
for w in attention block decode lstm; do
	PYTHONPATH=$site python3 tools/bench/attention.py "$w" 4 cuda | tee -a "$out/torch.txt"
done

echo "## mimir"
bench() {
	MIMIR_DEVICE=gpu MIMIR_CUDA_GRAPHS=1 go test -p 1 -run '^$' -count "$count" "$@" | tee -a "$out/mimir.txt"
}
bench -bench '^BenchmarkAttention$/^fused$' -benchtime 10x ./tensor
bench -bench '^BenchmarkDecoderBlockStep$/^seq[0-9]+$/^$' -benchtime 10x ./nn
bench -bench '^BenchmarkDecode$' -benchtime 20x ./nn
bench -bench '^BenchmarkLSTM$/^(forward|backward|train)$/^$' -benchtime 10x ./nn

echo "## M6 speed gates (mimir's time over PyTorch's, at most $gate)"
awk -v gate="$gate" '
	# PyTorch: "name: best 1.23 ms, median 1.45 ms".
	FILENAME ~ /torch/ {
		split($0, kv, ": ")
		for (i = 1; i <= NF; i++) if ($i == "median") torch[kv[1]] = $(i + 1) * 1e6
		next
	}
	# mimir: "BenchmarkX/sub-8  10  123456 ns/op ...", one line a run.
	/^Benchmark/ {
		name = $1
		sub(/-[0-9]+$/, "", name)
		for (i = 2; i < NF; i++) if ($(i + 1) == "ns/op") runs[name] = runs[name] " " $i
	}
	function median(s,    a, n, i, j, t) {
		n = split(s, a, " ")
		for (i = 2; i <= n; i++) for (j = i; j > 1 && a[j - 1] + 0 > a[j] + 0; j--) { t = a[j]; a[j] = a[j - 1]; a[j - 1] = t }
		return n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
	}
	END {
		pairs["attention seq256"] = "BenchmarkAttention/fused/seq256"
		pairs["attention seq1024"] = "BenchmarkAttention/fused/seq1024"
		pairs["block step seq256"] = "BenchmarkDecoderBlockStep/seq256"
		pairs["block step seq1024"] = "BenchmarkDecoderBlockStep/seq1024"
		pairs["decode attention"] = "BenchmarkDecode/attention"
		pairs["decode block"] = "BenchmarkDecode/block"
		pairs["lstm forward"] = "BenchmarkLSTM/forward"
		pairs["lstm backward"] = "BenchmarkLSTM/backward"
		pairs["lstm train"] = "BenchmarkLSTM/train"
		order = "attention seq256,attention seq1024,block step seq256,block step seq1024,decode attention,decode block,lstm forward,lstm backward,lstm train"
		n = split(order, names, ",")
		failed = 0
		for (i = 1; i <= n; i++) {
			w = names[i]
			if (!(w in torch) || !(pairs[w] in runs)) {
				printf "%-20s missing a result\n", w
				failed = 1
				continue
			}
			m = median(runs[pairs[w]])
			r = m / torch[w]
			verdict = r <= gate ? "pass" : "FAIL"
			if (r > gate) failed = 1
			printf "%-20s mimir %9.3f ms  PyTorch %9.3f ms  %5.2fx  %s\n", w, m / 1e6, torch[w] / 1e6, r, verdict
		}
		exit failed
	}
' "$out/torch.txt" "$out/mimir.txt"
