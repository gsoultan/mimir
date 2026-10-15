# mimir

mimir is a machine learning framework written in Go. It aims to train and run
neural networks as fast as PyTorch, on CPU and on GPU, from one pure-Go
codebase with no cgo and no Python.

> **Status: milestone 3.** mimir trains neural networks on the CPU with
> SIMD matrix kernels, fused elementwise kernels and in-place updates:
> autograd, layers, losses, optimizers, data loading and a training loop,
> each checked against PyTorch. The Metal GPU backend arrives in milestone 4.
> The API may change between 0.x releases.

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

## Requirements

Go 1.27 or later. The tensor API uses generic methods (`t.Data[float32](ctx)`).

## Goals

- **Fast on CPU and GPU.** One graph, one set of kernels per device, chosen at
  run time. Apple Metal first, then CUDA.
- **Never panic, never run out of memory.** Shape and dtype errors are
  values; kernels recover internal faults as `ErrInternal`; requests larger
  than the runtime can address return `ErrOutOfMemory`; intermediate buffers
  are freed as soon as their last reader finishes. The memory budget planner
  in milestone 5 will refuse work that does not fit in RAM before it starts.
- **Billion-row data.** Streaming datasets and memory budgets (milestone 5).
- **PyTorch parity.** Every operation is tested against PyTorch outputs, and
  random numbers are reproducible across devices (counter-based Philox).
- **Small core.** About 30 primitive operations. Everything else, including
  softmax, GELU and variance, is composed from them, so a new backend only has
  to implement the primitives.

## Non-goals

- A Python API or Python interop at run time.
- Dynamic dtype promotion. Mixing float32 and int32 is an error; cast first.
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
| Linear algebra | `MatMul` with NumPy semantics and batch broadcasting |
| Indexing | `Gather`, `ScatterAdd`, `Argmax` |
| Joining and windows | `Cat`, `Stack`, `Flatten`, `Unfold` |
| Convolution | `Conv2d` (stride, padding, dilation, groups), `MaxPool2d`, `AvgPool2d` |
| Autograd | `Backward`, `Grad`, `RequiresGrad`, `Detach`, `Param`, `Func` |
| Layers (`nn`) | `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `Embedding`, `Dropout`, `Sequential`, activations, pooling, `Flatten` |
| Losses (`nn/loss`) | `CrossEntropy`, `NLL`, `BCEWithLogits`, `MSE`, `L1`, `Huber` |
| Optimizers (`optim`) | `SGD` (momentum, Nesterov), `Adam`, `AdamW`, `RMSprop`; `StepDecay`, `Cosine`, `Warmup` schedules |
| Data (`data`, `data/datasets`) | batching, shuffling, MNIST, CSV |
| Training (`train`, `metrics`) | `Fit` with evaluation, schedules, early stopping, checkpoints and exact resume; `Accuracy` |

Element types are `float32` and `int32`. Broadcasting follows NumPy.
Modules are plain structs: `nn.Parameters` finds every `*tensor.Param` a model
holds, in exported or unexported fields, so custom layers need no
registration.

## Layout

```
tensor/              public Tensor API and autograd
nn/                  layers, parameter discovery, save and load
nn/loss/             loss functions
optim/               optimizers and learning-rate schedules
data/                datasets and batching
data/datasets/       MNIST and CSV
train/               Fit, evaluation, checkpoints
metrics/             accuracy
backend/             Backend interface, DType, Op, View, Kernel
backend/cpu/         pure-Go kernels and worker pool
internal/graph/      lazy IR with shape and dtype checks
internal/sched/      runs graphs: views, buffer reuse, freeing
internal/goldentest/ golden-file loading and comparison
io/safetensors/      safetensors reader and writer
examples/mnist/      trains an MLP or CNN on MNIST
testdata/golden/     expected results from PyTorch or NumPy
tools/bench/         PyTorch and NumPy counterparts of the benchmarks
tools/golden/        golden-file generator (Python, development only)
```

## Testing

```sh
go test -race ./...
MIMIR_EAGER=1 go test ./...
go test -run='^$' -fuzz=FuzzOps -fuzztime=60s ./tensor
```

`testdata/golden/` holds inputs and expected results: every operation's
output, the gradients of operations and layers, every layer and loss, and
optimizer steps and learning-rate schedules. The committed files were computed
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

## Roadmap

| Milestone | Delivers |
| --- | --- |
| M0 Foundation | module, CI, golden harness, benchmarks, fuzzing (done) |
| M1 Tensors that compute | lazy graph, CPU kernels, composite ops (done) |
| M2 Train MNIST | autograd, `nn` layers, losses, optimizers, data loading, `Fit`; first release v0.1.0 (done) |
| M3 Fast on CPU | SIMD GEMM microkernels with fused epilogues, fused elementwise kernels, convolution as GEMM, in-place optimizer updates with no allocations per step (done) |
| M4 Same code on GPU | Metal backend without cgo, unified memory, automatic device choice with CPU fallback |
| M4b CUDA | cuBLASLt, cuDNN and NVRTC through purego |
| M5 Billion-row scale | streaming Parquet and CSV shards, memory budget planner, memory-mapped data, bf16 and f16 |
| M6 Transformers and LLMs | attention and KV cache, RNN, LSTM and GRU, GGUF quantized matmul |
| M7 Full coverage | distributions, linear algebra, function transforms, ONNX import |

## License

Apache 2.0. See [LICENSE](LICENSE).
