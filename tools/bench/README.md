# Benchmarks

mimir's CPU benchmarks next to NumPy and PyTorch running the same work.
mimir reaches 77 to 99 percent of OpenBLAS on float32 matrix
multiplication, and PyTorch trains 1.4 to 2.1 times faster.

## Results

Measured on 2026-10-15 on a virtual machine with 4 cores of an Intel Xeon
at 2.10 GHz with AVX-512 (8 MiB of L2 cache), using Go 1.27.0, PyTorch
2.14.1, NumPy with OpenBLAS and NumPy with MKL, all on 4 threads. Other
work shares the machine, so times vary between runs. Each figure is the
best of at least three runs, with the median in parentheses.

### Matrix multiplication

Square float32 products, in GFLOP/s:

| n | mimir | OpenBLAS | MKL | mimir / OpenBLAS |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 93 (86) | 111 (109) | 98 (96) | 84% |
| 256 | 294 (278) | 297 (278) | 403 (382) | 99% |
| 512 | 364 (327) | 406 (396) | 451 (362) | 90% |
| 1024 | 410 (339) | 513 (506) | 540 (431) | 80% |
| 2048 | 405 (336) | 529 (517) | 514 (506) | 77% |

A mimir run reports the mean over its iterations, and a `gemm.py` run the
best of three timing loops, so the medians favor NumPy.

Apple Accelerate needs a Mac and is not measured here. NumPy's wheels use
it on macOS 14 or later, so the same commands on a Mac compare against it.

### Training steps

One training step: the forward pass, the cross-entropy loss, the backward
pass and an AdamW update, in milliseconds:

| Model | Batch | mimir | PyTorch | PyTorch faster by |
| --- | ---: | ---: | ---: | ---: |
| MNIST MLP | 128 | 1.9 (2.6) | 1.1 (1.3) | 1.8x (2.0x) |
| MNIST CNN | 128 | 38.9 (50.2) | 26.6 (35.4) | 1.5x (1.4x) |
| ResNet-18, 224×224 | 16 | 1199 (1275) | 571 (686) | 2.1x (1.9x) |

In mimir's ResNet-18 step, matrix products take 60 percent of the kernel
time, at 200 to 380 GFLOP/s. Copying windows for convolutions and adding
them back (`Unfold` and `Fold`) take 9 percent, and batch normalization and
other elementwise kernels most of the rest. PyTorch convolves directly with
oneDNN instead of copying windows.

### Deep MLP and bfloat16

A deep fully connected network for tables (20 → 1024 → 1024 → 1024 → 512 →
256 → 3 with ReLU), one training step on a batch of 512, measured on
2026-10-22 on the same machine. mimir reports the mean of its iterations
and PyTorch the median, with the best in parentheses:

| Precision | mimir | PyTorch |
| --- | ---: | ---: |
| float32 | 31.1 | 33.5 (28.2) |
| bfloat16 matmuls, float32 accumulate | 17.5 to 21.0 | 19.6 to 22.2 (15.9) |

The bfloat16 rows set `MIMIR_MATMUL_PRECISION=bf16` (or call
`cpu.SetMatMulBFloat16(true)`) for mimir and run under
`torch.autocast("cpu", torch.bfloat16)` for PyTorch. On CPUs with Intel
AMX, mimir then multiplies on AMX tiles; elsewhere the setting does
nothing. Training that network for 3 epochs on a 500,000-row table reached
99.2% accuracy in bfloat16 against 99.3% in float32, in 70 s instead of
111 s.

### Training on MNIST

The convolutional network of `examples/mnist`, trained for 10 epochs with
AdamW and cosine annealing:

| | Test accuracy | Time per epoch |
| --- | ---: | ---: |
| mimir | 99.14% | 24.9 s |
| PyTorch | 99.22% | 13.4 s |

## Running them

```sh
# mimir
go test -run='^$' -bench=SGEMM/avx512 -count=6 ./backend/cpu
MIMIR_BENCH_RESNET=1 go test -run='^$' -bench=TrainStep -benchtime=20x -count=6 ./train
MIMIR_MATMUL_PRECISION=bf16 go test -run='^$' -bench='TrainStep/deep$' -benchtime=200x -cpu=4 ./train
go run ./examples/mnist -model cnn -epochs 10

# NumPy and PyTorch, on as many threads as mimir uses
pip install numpy torch
OPENBLAS_NUM_THREADS=4 python3 tools/bench/gemm.py
python3 tools/bench/train_step.py mlp 4    # or cnn, resnet18, deep
python3 tools/bench/train_step.py deep 4 bf16
python3 tools/bench/mnist.py data/mnist 10
```

`python3 -c "import numpy; numpy.show_config()"` shows which BLAS NumPy
uses. BenchmarkSGEMM runs every GEMM kernel the CPU supports, and mimir
uses the first it lists: `avx512` here.
