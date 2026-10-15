# mimir

mimir is a machine learning framework written in Go. It aims to train and run
neural networks as fast as PyTorch, on CPU and on GPU, from one pure-Go
codebase with no cgo and no Python.

> **Status: milestones 0 and 1.** Tensors compute on the CPU and match PyTorch
> on every operation tested. Training (autograd, layers, optimizers) arrives in
> milestone 2, the Metal GPU backend in milestone 4. The API will change until
> v0.1.0.

## Example

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
| Indexing | `Gather`, `ScatterAdd` |

Element types are `float32` and `int32`. Broadcasting follows NumPy.

## Layout

```
tensor/              public Tensor API
backend/             Backend interface, DType, Op, View, Kernel
backend/cpu/         pure-Go kernels and worker pool
internal/graph/      lazy IR with shape and dtype checks
internal/sched/      runs graphs: views, buffer reuse, freeing
internal/goldentest/ golden-file loading and comparison
io/safetensors/      safetensors reader
tools/golden/        golden-file generator (Python, development only)
```

## Testing

```sh
go test -race ./...
MIMIR_EAGER=1 go test ./tensor/...
go test -run='^$' -fuzz=FuzzOps -fuzztime=60s ./tensor
```

`tensor/testdata/golden/ops.safetensors` holds inputs and expected outputs for
every operation, plus a layer norm, an attention block and a two-layer MLP. The
committed file was computed with NumPy in float64. CI regenerates it with
PyTorch and runs the same tests against it:

```sh
pip install numpy torch
python3 tools/golden/gen.py --oracle torch --out /tmp/golden
MIMIR_GOLDEN_DIR=/tmp/golden go test -run TestGolden ./tensor
```

## Roadmap

| Milestone | Delivers |
| --- | --- |
| M0 Foundation | module, CI, golden harness, benchmarks, fuzzing (done) |
| M1 Tensors that compute | lazy graph, CPU kernels, composite ops (done) |
| M2 Train MNIST | autograd, `nn` layers, losses, optimizers, data loading, `Fit`; first release v0.1.0 |
| M3 Fast on CPU | SIMD GEMM microkernels, fused elementwise kernels, arena allocator, schedule cache |
| M4 Same code on GPU | Metal backend without cgo, unified memory, automatic device choice with CPU fallback |
| M4b CUDA | cuBLASLt, cuDNN and NVRTC through purego |
| M5 Billion-row scale | streaming Parquet and CSV shards, memory budget planner, memory-mapped data, bf16 and f16 |
| M6 Transformers and LLMs | attention and KV cache, RNN, LSTM and GRU, GGUF quantized matmul |
| M7 Full coverage | distributions, linear algebra, function transforms, ONNX import |

## License

Apache 2.0. See [LICENSE](LICENSE).
