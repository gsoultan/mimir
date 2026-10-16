# mimir

mimir is a machine learning framework written in Go. It aims to train and run
neural networks as fast as PyTorch, on CPU and on GPU, from one pure-Go
codebase with no cgo and no Python.

> **Status: milestone 6.** mimir trains neural networks on the CPU with
> SIMD matrix kernels, fused elementwise kernels and in-place updates, and
> runs the same code on Apple GPUs through Metal and on NVIDIA GPUs through
> CUDA: autograd, layers, losses, optimizers, data loading and a training
> loop, each checked against PyTorch. It streams training sets larger than
> memory from Parquet and CSV shards in flat memory, plans every
> computation against a memory budget, stores tensors in bfloat16 and
> float16, and replays repeated training steps from a cached schedule. It
> has recurrent and transformer layers with fused attention, and runs
> quantized Llama models from GGUF files. The API may change between 0.x
> releases.

## Example

Train a convolutional network on MNIST:

```go
model := nn.NewSequential(
	nn.NewConv2d(1, 32, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
	nn.NewConv2d(32, 64, 3, nn.Padding(1)), nn.ReLU(), nn.MaxPool2d(2),
	nn.Flatten(), nn.NewLinear(64*7*7, 128), nn.ReLU(), nn.NewLinear(128, 10),
)
opt := optim.AdamW(nn.Parameters(model), optim.LR(1e-3), optim.WeightDecay(0.01))
trainSet, testSet, err := datasets.MNIST("data/mnist", data.BatchSize(128), data.Shuffle(42))
if err != nil {
	return err
}
err = train.Fit(ctx, model, opt, trainSet,
	train.Loss(loss.CrossEntropy),
	train.Epochs(10),
	train.Schedule(optim.Cosine(1e-3, 0, 10)),
	train.Eval(testSet, metrics.Accuracy),
	train.Checkpoint("ckpt/"),
	train.Progress(os.Stdout))
```

`go run ./examples/mnist -model cnn -epochs 10` runs it, downloading MNIST
first. On four CPU cores this network reaches 99.1% test accuracy in 10
epochs of about 25 seconds each, and `-model mlp` reaches 96.6% in three
epochs of about 1.2 seconds. Or write the loop yourself:

```go
for batch, err := range trainSet.All(ctx) {
	if err != nil {
		return err
	}
	l := loss.CrossEntropy(model.Forward(batch.X), batch.Y)
	if err := l.Backward(ctx); err != nil { // loss and gradients in one pass
		return err
	}
	if err := opt.Step(ctx); err != nil {
		return err
	}
	opt.ZeroGrad()
}
```

Tensors on their own:

```go
ctx := context.Background()
x := tensor.Randn(1, 4, 8)   // batch of 4, 8 features
w1 := tensor.Randn(2, 8, 16) // seeded: identical on every machine
w2 := tensor.Randn(3, 16, 3)

probs := x.MatMul(w1).GELU().MatMul(w2).Softmax(-1) // builds a graph
out, err := probs.Data[float32](ctx)                 // computes it
```

Operations are lazy: each call checks shapes and records a node, and nothing
runs until `Realize`, `Data` or `Item`. Nothing panics. A bad operation returns
a tensor carrying the error, everything built on it carries the same error, and
`Err` or `Data` reports it:

```go
y := a.MatMul(b).ReLU()           // a is 2x3, b is 4x5
errors.Is(y.Err(), tensor.ErrShape) // true
```

Set `MIMIR_EAGER=1` to compute every operation as soon as it is created, so a
runtime error such as an out-of-range index surfaces at the line that caused
it.

## Devices

The same program runs on the CPU or the GPU. `mimir.Auto` returns the GPU if
there is one mimir can use and the CPU otherwise, so code written for a Mac
still runs on a Linux server:

```go
dev := mimir.Auto()       // or mimir.CPU(), or mimir.GPU()
err := nn.To(model, dev)  // parameters, buffers and gradients
x = x.To(dev)             // a tensor; gradients flow back through the copy
```

`mimir.Devices` lists every device available. `train.Fit` moves each batch
to the device the model is on. Tensors that `FromSlice` creates live on the
default device, which `MIMIR_DEVICE` sets to `cpu`, `gpu` or `auto`; combining
tensors on different devices returns `tensor.ErrDevice`. `go run
./examples/mnist -device auto` trains on the GPU when there is one.

On Apple silicon the GPU runs through Metal, and on Linux an NVIDIA GPU runs
through CUDA, both called from Go with
[purego](https://github.com/ebitengine/purego), so there is still no cgo and
the binary cross-compiles. One generator writes every kernel in Metal Shading
Language or CUDA C++, compiled at run time once per kernel structure (CUDA
kernels are cached on disk), and float32 matrix products run on Metal
Performance Shaders, on Metal 4 tensor operations on M5 GPUs under macOS 26,
or on cuBLAS. On a Mac, CPU and GPU share memory, so moving a tensor is a
copy within RAM; on CUDA, device memory comes from a pool that keeps freed
buffers, so a training step allocates nothing from the driver once it has
run. Every operation runs the same conformance tests on every device, and
the generated kernels of both dialects are also compiled and run on the CPU
in CI.

CUDA needs the NVIDIA driver and `libcuda.so.1`, `libnvrtc.so.12` and
`libcublas.so.12`, as the `nvidia/cuda` runtime images and the CUDA toolkit
install them. Loading them needs the C library, so on Linux a program that
uses the `tensor` package is linked dynamically even with `CGO_ENABLED=0`;
`go build -tags nocuda` leaves the backend out and keeps the binary static.

## Gradients

Tensors from `RequiresGrad` or a `tensor.Param` require gradients, and so does
everything computed from them. `Backward` on a one-element loss computes every
gradient and the loss itself in one schedule, so work the forward and backward
passes share runs once, then releases the graph:

```go
w := tensor.Randn(1, 3, 2).RequiresGrad()
loss := x.MatMul(w).Sum()
err := loss.Backward(ctx)
dw := w.Grad() // shaped like w
```

Every differentiable operation has a gradient, derived from about twenty
rules for the primitives; `tensor.Func` adds hand-written ones.

Parameters (`tensor.Param`) are the only values that change. An optimizer step
writes the new parameter values and optimizer state over the old ones, in
place like PyTorch, and `Backward` accumulates into existing gradients the same
way, so a training step reuses the same memory every time. A tensor taken from
a parameter before a step is stale after it: reading it, or computing anything
built from it, returns `tensor.ErrOverwritten`. Realize what you read after a
step, such as predictions, before it, or realize a `Clone` to keep a copy.
`tensor.RealizeInto` writes your own results over existing tensors.

Function transforms in the manner of JAX work on top of the same graph.
`tensor.Grad`, `VJP` and `JVP` give gradients, vector-Jacobian and
Jacobian-vector products of a Go function; `Jacobian` and `Hessian` build on
them. Their gradients are lazy and themselves differentiable
(`tensor.Gradients`), so `JVP` of `Grad` is a Hessian-vector product. `VMap`
runs a function over a batch axis without a loop, which gives per-example
gradients in one pass, about twelve times as fast as looping over a batch of
64:

```go
loss := func(x ...*tensor.Tensor) *tensor.Tensor { return x[0].MatMul(w).Tanh().Sum() }
perExample := tensor.VMap(func(x ...*tensor.Tensor) []*tensor.Tensor {
	return tensor.Gradients(loss(x...), w)
})(batch) // one gradient of w for each row of batch
```

An operation without a rule returns `tensor.ErrNotDifferentiable` or
`tensor.ErrNotBatchable` rather than a wrong result: the factorizations have
no batching or second derivatives yet, and random operations inside `VMap`
draw the same values for every example.

## Performance

On the CPU, mimir multiplies float32 matrices at 77 to 99 percent of the
speed of OpenBLAS, with assembly microkernels for AVX2, AVX-512 and NEON.
Chains of elementwise operations run as one fused kernel, and so do a
matrix product and the bias, activation or residual applied to it. Once
the first steps have run, a training step reuses its memory and allocates
only the loss it returns and the running statistics of batch normalization.
PyTorch still trains 1.4 to 2.1 times faster: see
[tools/bench](tools/bench/README.md) for the measurements and how to
repeat them.

On an Apple GPU, Metal Performance Shaders multiplies 2048 × 2048 float32
matrices at about 1,380 GFLOP/s on the M1 virtual machine GitHub's macOS
runners provide, against about 100 GFLOP/s on its three CPU cores. Training
the small MNIST networks is still slower there than on the CPU: a step of
the MLP takes 4.4 ms on the GPU and 3.4 ms on the CPU, and the CNN 128 ms
against 80 ms, because each step runs many small kernels and waits for the
GPU to finish. The CI job `gpu` prints these numbers for every change, and
the job `cuda` prints the same numbers for an NVIDIA T4 when the repository
has Modal credentials (see Testing).

A training step builds the same graph every time, so mimir compiles it once.
The scheduler keys each schedule on the graph's structure (operations,
shapes, dtypes, devices and which tensors are realized) and, when a step
repeats, replays the planned kernels into the same buffers, passing changed
constants and random seeds as kernel arguments, without building, sorting
or planning anything or allocating. On CUDA the replayed kernels and cuBLAS
calls run as one CUDA graph from the second step, updated in place when an
argument changes (`MIMIR_CUDA_GRAPHS=0` launches them one by one instead),
and on Metal they are encoded into one command buffer. On the GPUs,
convolutions run as tiled implicit-GEMM kernels that read their windows
from the image where they lie, for the forward pass and both gradients;
the CPU multiplies unfolded windows with its GEMM microkernels.

On an M5 Pro under macOS 26, the GPU's neural accelerators run float32
products through Metal 4 tensor operations at 5,790 GFLOP/s for 2048 × 2048
and 6,050 for 4096 × 4096, 3 to 11% faster than Metal Performance Shaders
at every size measured and with the same results bit for bit, against
about 940 GFLOP/s on its CPU cores. Older GPUs and systems keep using MPS.

## Scale

A training set larger than memory streams from shards: the row groups of
Parquet files, CSV files, or any `data.Source`. A `data.Stream` reads a few
shards ahead of the batches it yields, shuffles the shard order and a
buffer of examples every epoch, and holds the same few megabytes whatever
the number of rows. `data.Statistics` reads the source once, in parallel,
for the per-feature mean and deviation that `data.Normalize` applies:

```go
src, features, err := datasets.ParquetShards(paths, "label", datasets.Classes("no", "yes"))
if err != nil {
	return err
}
stats, err := data.Statistics(ctx, src)
if err != nil {
	return err
}
trainSet, err := data.NewStream(src, data.BatchSize(4096), data.Shuffle(42),
	data.ClassTargets(), data.Normalize(stats))
```

CI trains a small network over a billion synthetic rows on every change.
On a GitHub-hosted Linux runner it takes about 19 minutes, 895,000 rows a
second, and the process stays at 32 MiB resident from the first tenth of
the epoch to the last.

Memory is planned before it is spent. `tensor.SetMemoryBudget` caps what
computations on a device may hold: a schedule that would go past it fails
with `tensor.ErrMemoryBudget` before it allocates or computes anything.
`train.MemoryBudget` trains within such a budget by splitting each batch
that does not fit into smaller chunks and accumulating their gradients,
and `train.MicroBatch` does the same with a fixed chunk size.
`nn.Checkpoint` and `tensor.Checkpoint` keep only the inputs of a block
during the forward pass and recompute the rest in `Backward`, trading
compute for activation memory.

CPU buffers of a megabyte or more live in anonymous memory mappings
outside the Go heap, returned to the operating system as soon as they are
freed. `safetensors.Open` maps a file instead of reading it, so
`nn.Load` and `datasets.Mapped` read weights and examples in place, and the
file's pages stream in from disk as they are used.

Tensors can be stored as `tensor.BFloat16` or `tensor.Float16`, which
halves their memory; operations compute in float32 and round on the way
out. `nn.Cast` converts a model, `optim.MasterWeights` keeps a float32
copy of each half parameter for the optimizer to update, and
`train.LossScale` or `train.DynamicLossScale` scale the loss so small
gradients survive float16.

## Sequences, transformers and LLMs

`tensor.Scan` runs a step function over an axis, like `jax.lax.scan`, taking
a view per step and stacking the outputs once, in either direction and
optionally recomputing each step in `Backward`. `CumSum`, `CumMax` and
`LogCumSumExp` are single kernels on every device. `nn.RNN`, `nn.LSTM` and
`nn.GRU` follow PyTorch's layouts and options (layers, bidirectional,
`BatchFirst`, dropout between layers) and match it to float32 precision;
an LSTM step is one matrix product with the gate math fused into it.

`tensor.ScaledDotProductAttention` computes attention in one fused kernel
that never materializes the scores, forward and backward, as tiled
FlashAttention on the CPU and generated kernels on Metal and CUDA, with
causal, additive and boolean masks and grouped-query attention. On four CPU
cores it trains at the speed of PyTorch's attention at sequence length 1024
and 5 to 15 times faster than composing it from products and a softmax.
`nn` builds transformers from it: `MultiheadAttention`, `SelfAttention`
with fused QKV and rotary embeddings (`RotaryEmbedding`, with Llama 3
scaling), `RMSNorm`, `MLP`, `GatedMLP`, `KVCache` for decoding, the
`DecoderBlock`s of `NewGPT2Block` and `NewLlamaBlock`, and
`TransformerEncoderLayer`.

`gguf.Open` maps a GGUF file and `File.Load` returns its tensors, including
the block-quantized `tensor.Q8_0`, `Q4_K` and `Q6_K`. A quantized weight
stays quantized: `MatMulT`, and so `nn.Linear` and `nn.Embedding`, multiply
by it as llama.cpp does, rounding activations to int8 and summing in
integers with AVX2 or NEON dot-product kernels, and GPUs dequantize it in
the kernel. Quantized weights are for inference and have no gradient. A
one-row product by a Q4_K weight takes 0.7 to 0.9 times llama.cpp's time on
the same CPU, and products of many rows, as a prompt needs, run in tiles
with AVX-512 VNNI, AVX2 or SDOT kernels chosen at run time. On one core,
a Llama-3.2-1B-shaped Q4_K_M model decodes at about 1.15 times llama.cpp's
time per token and reads a 64-token prompt at about 1.35 times.

`examples/gpt2` trains a byte-level GPT-2 from scratch on a text file, with
warmup and cosine decay, gradient clipping and bfloat16 on a GPU, and
samples from it with the KV cache:

```sh
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
go run ./examples/gpt2 -data input.txt -model small -steps 2000 -checkpoint gpt2.safetensors
go run ./examples/gpt2 -generate -checkpoint gpt2.safetensors -prompt "ROMEO:" -temperature 0.8 -topk 40
```

`examples/llama` runs Llama-architecture GGUF models (Llama 2 and 3,
TinyLlama, SmolLM) in their own quantization, with the file's SentencePiece
or BPE tokenizer, a batched prompt pass and greedy, temperature, top-k or
top-p decoding:

```sh
go run ./examples/llama -model Llama-3.2-1B-Instruct-Q4_K_M.gguf -prompt "The capital of France is" -n 64 -temperature 0
```

## Requirements

Go 1.27 or later. The tensor API uses generic methods (`t.Data[float32](ctx)`).
mimir depends on two modules: [purego](https://github.com/ebitengine/purego)
to call Metal and CUDA without cgo, and
[klauspost/compress](https://github.com/klauspost/compress) for the
Snappy and zstd pages of Parquet files, which the standard library cannot
read.

## Goals

- **Fast on CPU and GPU.** One graph, one set of kernels per device, chosen at
  run time: Apple Metal and NVIDIA CUDA today.
- **Never panic, never run out of memory.** Shape and dtype errors are
  values; kernels recover internal faults as `ErrInternal`; requests larger
  than the runtime can address return `ErrOutOfMemory`; intermediate buffers
  are freed as soon as their last reader finishes; `tensor.SetMemoryBudget`
  refuses work that does not fit before it starts.
- **Billion-row data.** Training sets stream from Parquet and CSV shards in
  flat memory, and CI trains over a billion rows on every change.
- **PyTorch parity.** Every operation is tested against PyTorch outputs, and
  random numbers are reproducible across devices (counter-based Philox).
- **Small core.** About 30 primitive operations. Everything else, including
  softmax, GELU and variance, is composed from them, so a new backend only has
  to implement the primitives. A few fused operations, such as attention,
  scans and quantized products, have their own kernels because composing
  them would be much slower.

## Non-goals

- A Python API or Python interop at run time.
- Dynamic dtype promotion. Mixing float32 and int32 is an error; cast first.
  The one exception is the half types: an operation on bfloat16 or float16
  and float32 computes in float32.
- Matching every PyTorch operator name. mimir follows PyTorch semantics where
  it has the same operation.

## What works today

| Area | Operations |
| --- | --- |
| Create | `FromSlice`, `Full`, `Zeros`, `Ones`, `Scalar`, `Arange`, `Rand`, `Randn` |
| Elementwise | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Exp`, `Exp2`, `Log`, `Log2`, `Sin`, `Cos`, `Sqrt`, `Recip`, `Abs`, `Maximum`, `Minimum`, `Less`, `Greater`, `Equal`, `Fmod`, `Where`, `Cast` |
| Activations | `ReLU`, `Sigmoid`, `Tanh`, `GELU` (tanh approximation), `SiLU`, `Softmax`, `LogSoftmax` |
| Reductions | `Sum`, `Max`, `Min`, `Mean`, `Var` |
| Shape | `Reshape`, `Permute`, `Transpose`, `Expand`, `Pad`, `Shrink`, `Flip`, `Unsqueeze`, `Squeeze` |
| Linear algebra | `MatMul` with NumPy semantics and batch broadcasting, `MatMulT` (also by quantized weights) |
| Indexing | `Gather`, `ScatterAdd`, `Argmax` |
| Joining and windows | `Cat`, `Stack`, `Flatten`, `Unfold`, `SliceScatter` |
| Scans | `CumSum`, `CumMax`, `LogCumSumExp`, `Scan` |
| Attention | `ScaledDotProductAttention` (causal, masks, grouped-query) |
| Sorting | `Sort`, `Argsort`, `TopK` (stable; NaN sorts last) |
| Fourier | `FFT`, `IFFT`, `RFFT`, `IRFFT` (any length; complex numbers as a trailing axis of 2) |
| Linear algebra | `Cholesky`, `TriangularSolve`, `Solve`, `Inv`, `Det`, `SlogDet`, `QR`, `SVD`, `Eigh` (batched, with gradients; on the CPU, so GPU tensors make a round trip) |
| Convolution | `Conv2d` (stride, padding, dilation, groups), `MaxPool2d`, `AvgPool2d` |
| Autograd | `Backward`, `Grad`, `RequiresGrad`, `Detach`, `Param`, `Func`, `Checkpoint` |
| Memory | `SetMemoryBudget`, `ErrMemoryBudget`, `FromBytes` over mapped files |
| Layers (`nn`) | `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`, `Sequential`, `Checkpoint`, activations, pooling, `Flatten`, `Cast` |
| Sequence layers (`nn`) | `RNN`, `LSTM`, `GRU`, `MultiheadAttention`, `SelfAttention`, `RotaryEmbedding`, `KVCache`, `MLP`, `GatedMLP`, `DecoderBlock`, `TransformerEncoderLayer` |
| Losses (`nn/loss`) | `CrossEntropy`, `NLL`, `BCEWithLogits`, `MSE`, `L1`, `Huber` |
| Optimizers (`optim`) | `SGD` (momentum, Nesterov), `Adam`, `AdamW`, `RMSprop`, `MasterWeights`; `StepDecay`, `Cosine`, `Warmup` schedules |
| Data (`data`, `data/datasets`) | batching, shuffling, `Stream`, `Statistics`, `Normalize`; MNIST, CSV, `CSVShards`, `ParquetShards`, `Mapped`, `Synthetic` |
| Files (`io`) | safetensors read, write and map; Parquet; GGUF read (F32, F16, BF16, Q8_0, Q4_K, Q6_K) and write |
| Training (`train`, `metrics`) | `Fit` with evaluation, schedules, early stopping, checkpoints and exact resume, `MicroBatch`, `MemoryBudget`, `LossScale`, `DynamicLossScale`; `Accuracy` |

Element types are `float32`, `int32`, `bfloat16` and `float16`, plus the
block-quantized `Q8_0`, `Q4_K` and `Q6_K` for weights. Broadcasting follows NumPy.
Modules are plain structs: `nn.Parameters` finds every `*tensor.Param` a model
holds, in exported or unexported fields, so custom layers need no
registration.

## Layout

```
tensor/              public Tensor API and autograd
nn/                  layers, parameter discovery, save and load
nn/loss/             loss functions
optim/               optimizers and learning-rate schedules
data/                datasets, batching, streams of shards and their statistics
data/datasets/       MNIST, CSV, CSV and Parquet shards, mapped safetensors, synthetic rows
train/               Fit, evaluation, checkpoints
metrics/             accuracy
backend/             Backend interface, DType, Op, View, Kernel
backend/cpu/         pure-Go kernels and worker pool
backend/metal/       Metal backend: Metal Performance Shaders and tensor operations, through purego
backend/cuda/        CUDA backend: driver API, NVRTC and cuBLAS, through purego
backend/internal/gen/        generates the kernels, in Metal Shading Language or CUDA C++
backend/internal/kerneltest/ kernel cases for every device, and a CPU emulator for both dialects
internal/graph/      lazy IR with shape and dtype checks
internal/sched/      runs graphs: views, buffer reuse, freeing, the schedule cache
internal/filemap/    read-only file mappings
internal/goldentest/ golden-file loading and comparison
internal/quanttest/  reference quantization for tests
io/safetensors/      safetensors reader, writer and file mapping
io/parquet/          Parquet reader: flat schemas, every standard encoding, Snappy, gzip and zstd
io/gguf/             GGUF reader and writer
examples/mnist/      trains an MLP or CNN on MNIST
examples/gpt2/       trains a GPT-2 on a text file and samples from it
examples/llama/      runs Llama-architecture GGUF models
testdata/golden/     expected results from PyTorch or NumPy
tools/asmgen/        generates the CPU assembly kernels with avo
tools/devices/       lists the devices mimir can use
tools/bench/         PyTorch and NumPy counterparts of the benchmarks
tools/golden/        golden-file generator (Python, development only)
tools/parquet/       Parquet test-file generator (Python, development only)
tools/gguf/          GGUF test-file generators (Python, development only)
```

## Testing

```sh
go test -race ./...
MIMIR_EAGER=1 go test ./...
MIMIR_DEVICE=gpu go test ./...   # on a Mac or an NVIDIA machine: every test on the GPU
MIMIR_EMULATE=1 go test -timeout=60m -run Kernels ./backend/...   # the GPU kernels on the CPU, with clang++
go test -run='^$' -fuzz=FuzzOps -fuzztime=60s ./tensor
go test -run='^$' -fuzz=FuzzOpen -fuzztime=60s -fuzzminimizetime=0 ./io/parquet
go test -run='^$' -fuzz=FuzzOpen -fuzztime=60s ./io/gguf
MIMIR_SCALE_ROWS=1000000000 go test -run=FlatMemory -timeout=85m -v ./train   # the billion-row run
```

The scale test trains over a million rows by default; `MIMIR_SCALE_ROWS`
sets another number. The Parquet test files in `io/parquet/testdata` and
`data/datasets/testdata` were written by pyarrow; `python3
tools/parquet/gen.py` writes them again.

`testdata/golden/` holds inputs and expected results: every operation's
output, the gradients of operations and layers, every layer and loss, and
optimizer steps and learning-rate schedules, scans, recurrent and
transformer layers, and quantized weights. The committed files were computed
with NumPy in float64, with gradients by central differences. CI regenerates
them with PyTorch, using autograd in float64 for gradients, and runs the same
tests against them:

```sh
pip install numpy torch
python3 tools/golden/gen.py --oracle torch --out /tmp/golden
MIMIR_GOLDEN_DIR=/tmp/golden go test -run TestGolden ./...
```

`TestGradientsMatchFiniteDifferences` also checks every operation's gradient
against finite differences in Go.

GitHub has Apple silicon runners but no free NVIDIA ones, so the CUDA tests
run on a [Modal](https://modal.com) T4 through `ci/cuda/modal_test.py`, which
copies the checkout into a CUDA runtime container and runs `ci/cuda/test.sh`
there. The CI job `cuda` runs it when the repository has the secrets
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, and skips with a warning
otherwise. `pip install modal && modal run ci/cuda/modal_test.py` runs it
from a laptop, and `sh ci/cuda/test.sh` runs the same tests on any Linux
machine with an NVIDIA GPU. The CUDA kernels are also compiled with the real
NVRTC on every change, which needs no GPU.

## Roadmap

| Milestone | Delivers |
| --- | --- |
| M0 Foundation | module, CI, golden harness, benchmarks, fuzzing (done) |
| M1 Tensors that compute | lazy graph, CPU kernels, composite ops (done) |
| M2 Train MNIST | autograd, `nn` layers, losses, optimizers, data loading, `Fit`; first release v0.1.0 (done) |
| M3 Fast on CPU | SIMD GEMM microkernels with fused epilogues, fused elementwise kernels, convolution as GEMM, in-place optimizer updates with no allocations per step (done) |
| M4 Same code on GPU | Metal backend without cgo, unified memory, MPS matrix products, fused kernels, automatic device choice with CPU fallback and Metal 4 tensor operations on M5 GPUs (done) |
| M4b CUDA | CUDA driver, NVRTC and cuBLAS through purego, stream-ordered memory pool, on-disk kernel cache, conformance tests on a Modal T4 (done) |
| M5 Billion-row scale | streaming Parquet and CSV shards, memory budget planner, micro-batching and activation checkpointing, off-heap and memory-mapped tensors, bf16 and f16 with loss scaling, a billion-row test in CI (done) |
| M5b Compiled steps | schedule cache, replay into fixed buffers with no allocations, CUDA graphs, one Metal command buffer per step (done) |
| M6 Transformers and LLMs | scans, RNN, LSTM and GRU, fused attention, transformer layers and KV cache, GGUF and quantized matmul, GPU convolution kernels, GPT-2 and Llama examples (in progress: speed gates) |
| M7 Full coverage | distributions and generative models, linear algebra, differentiable gradients, forward mode and vmap, ONNX import, ROCm (in progress: function transforms, sorting, FFT and linear algebra done) |
| M8 Multiple devices | data parallelism over NCCL, then tensor sharding modeled on `jax.sharding` |

## License

Apache 2.0. See [LICENSE](LICENSE).
