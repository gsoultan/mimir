# mimir

mimir is a machine learning framework written in Go. It aims to train and run
neural networks as fast as PyTorch, on CPU and on GPU, from one pure-Go
codebase with no cgo and no Python.

> **Status: milestone 8.** mimir trains neural networks on the CPU with
> SIMD matrix kernels, fused elementwise kernels and in-place updates, and
> runs the same code on Apple GPUs through Metal, on NVIDIA GPUs through
> CUDA and on AMD GPUs through ROCm: autograd, layers, losses, optimizers,
> data loading and a training loop, each checked against PyTorch. It trains
> over several devices at once, data-parallel or sharded over a mesh in
> the manner of `jax.sharding`. It streams training sets larger than
> memory from Parquet and CSV shards in flat memory, plans every
> computation against a memory budget, stores tensors in bfloat16 and
> float16, and replays repeated training steps from a cached schedule. It
> has recurrent and transformer layers with fused attention, and runs
> quantized Llama models from GGUF files. Function transforms, probability
> distributions, graph network layers and an ONNX interpreter build on the
> same graph. The API may change between 0.x releases.

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
through CUDA and an AMD GPU through ROCm's HIP, all called from Go with
[purego](https://github.com/ebitengine/purego), so there is still no cgo and
the binary cross-compiles. `gpu` takes Metal, then CUDA, then ROCm, and
`MIMIR_DEVICE=metal`, `cuda` or `rocm` opens only that backend, so `rocm`
uses an AMD GPU even beside an NVIDIA one; when there is no GPU, the error
says why for each backend the platform could have. One generator writes
every kernel in Metal Shading Language, CUDA C++ or HIP C++, the last for
AMD wavefronts of 64 or 32 threads, compiled at run time once per kernel
structure (CUDA and HIP kernels are cached on disk), and float32 matrix
products run on Metal Performance Shaders, on Metal 4 tensor operations on
M5 GPUs under macOS 26, on cuBLAS, or on hipBLAS. On a Mac, CPU and GPU
share memory, so moving a tensor is a copy within RAM; on CUDA and ROCm,
device memory comes from a pool that keeps freed buffers, so a training
step allocates nothing from the driver once it has run. Every operation
runs the same conformance tests on every device, and the generated kernels
of all four dialects are also compiled and run on the CPU in CI.

CUDA needs the NVIDIA driver and `libcuda.so.1`, `libnvrtc.so.12` and
`libcublas.so.12`, as the `nvidia/cuda` runtime images and the CUDA toolkit
install them. ROCm needs ROCm 5.4 or later, `libamdhip64` and hiprtc, as the
`rocm/dev` images and the ROCm packages install them; hipBLAS is optional,
and without it products run on the generated kernel. `MIMIR_HIP`,
`MIMIR_HIPRTC` and `MIMIR_HIPBLAS` name the libraries where the loader would
not find them. Loading them needs the C library, so on Linux a program that
uses the `tensor` package is linked dynamically even with `CGO_ENABLED=0`;
`go build -tags nocuda,norocm` leaves both backends out and keeps the binary
static.

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
or planning anything or allocating, and gives each kernel the buffers it
had in the last run. On Metal the replayed kernels are encoded into one
command buffer. On CUDA a step's kernels and cuBLAS calls run as one CUDA
graph, captured once a run has repeated the last one's products and
updated in place when only arguments change (`MIMIR_CUDA_GRAPHS=0` launches
them one by one instead); `MIMIR_ROCM_GRAPHS=1` does the same with HIP
graphs on ROCm, where they stay off until an AMD GPU measures them. On the GPUs, convolutions run
as tiled implicit-GEMM kernels that read their windows from the image
where they lie, for the forward pass and both gradients; the CPU
multiplies unfolded windows with its GEMM microkernels. On CUDA and ROCm, a
convolution too deep for its image to fill the device with tiles, such as
a 512-channel layer over a small image, runs as unfold, cuBLAS product and
fold instead, and products summed over batch axes or whose operand every
batch shares run on cuBLAS or hipBLAS too.

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

A batch can be donated to the step that reads it, as `jax.jit`'s
`donate_argnums` does. `Donate` marks a realized tensor as done with: the
next schedule that reads it gives its buffer back to the pool after its
last reader, and `FromSlice` takes a free buffer of its size before it
allocates, so a loop that uploads and donates its batches runs over the
same memory every step. Reading the tensor afterwards returns
`tensor.ErrDonated`, which wraps `ErrOverwritten`. `train.Fit` donates the
copy of each batch it moves to the model's device, and `examples/gpt2`
donates its batches:

```go
x := tensor.FromSlice(pixels, 128, 784).To(dev).Donate()
```

CPU buffers of a megabyte or more live in anonymous memory mappings
outside the Go heap. A freed buffer leaves its mapping in a bounded cache
that the next allocation of a similar size takes, clearing only the bytes
it needs, so an uploaded batch lands in pages already resident rather
than faulting in new ones; mappings that no allocation takes within two
garbage collections go back to the operating system, and
`cpu.CachedMapBytes` reports what the cache holds. Mapped buffers barely
grow the Go heap, so the CPU backend runs a collection itself once enough
has been mapped since the last, paced as `GOGC=50` would, and waits for
the cleanups that give dropped buffers' mappings back.

`safetensors.Open` maps a file instead of reading it, so `nn.Load` and
`datasets.Mapped` read weights and examples in place, and the file's pages
stream in from disk as they are used.

Tensors can be stored as `tensor.BFloat16` or `tensor.Float16`, which
halves their memory; operations compute in float32 and round on the way
out. `nn.Cast` converts a model, `optim.MasterWeights` keeps a float32
copy of each half parameter for the optimizer to update, and
`train.LossScale` or `train.DynamicLossScale` scale the loss so small
gradients survive float16.

## Multiple devices

`tensor.GPUs` returns every GPU mimir can use, and `tensor.CPUs(n)` returns
n CPU devices that share the host's cores, each running its kernels in
order on a goroutine of its own, as a GPU runs them on a stream.
`train.DataParallel` trains on all of them at once:

```go
err = train.Fit(ctx, model, opt, trainSet,
	train.Loss(loss.CrossEntropy),
	train.Epochs(10),
	train.DataParallel(tensor.GPUs()...))
```

`Fit` moves the model to the first device, replicates it onto the others
with `nn.Replicate`, and splits each batch between them along its first
axis. Each replica computes the loss and gradients of its share, the
gradients are all-reduced into every replica, weighted by each share's
part of the batch, and every device steps its own copy of the parameters
with the optimizer, so every replica holds the same values bit for bit
without copying them from the first device. Evaluation and checkpoints
use the model on the first device. `train.NewParallel` does the same in a
loop you write: its `Backward` takes a batch and leaves the whole batch's
gradients on the model, as PyTorch's `DistributedDataParallel` does within
one process, and its `Step` steps the optimizer on every device. Two and
four CPU devices match one device over the whole batch. `DataParallel` does not combine with `MicroBatch` or `MemoryBudget`.

Underneath, `tensor.NewGroup` connects devices of one kind, the CPU
devices or CUDA GPUs, like a group of NCCL ranks in one process. Its
`AllReduce`, `Broadcast`, `AllGather` and `ReduceScatter` take one tensor
per device and return realized tensors on every device, and
`ReduceGrads`, `AllReduceGrads` and `BroadcastParams` work on parameters in place, in
buckets of about 25 MiB as `DistributedDataParallel` does. CUDA devices
communicate through NCCL, which needs `libnccl.so.2`, and CPU devices
through host memory. The devices of a group run asynchronously: the host
queues a step on each without waiting for the last, so a kernel's error
surfaces at the next read of a value from its device.

## Sharding

Sharding is modeled on `jax.sharding`. `tensor.NewMesh` arranges devices in
n dimensions with named axes, and `tensor.Shard` places a tensor on the mesh
as a `PartitionSpec` says, each device holding its block. `tensor.P` builds
one, naming the mesh axis that splits each axis of the tensor, and
`WithSections` cuts an axis into sections, such as the query, key and value
of a fused projection, before it is split:

```go
mesh, err := tensor.NewMesh([]int{2, 2}, []string{"data", "model"}, tensor.CPUs(4)...)
if err != nil {
	return err
}
defer mesh.Close()
x := tensor.Shard(batch, mesh, tensor.P("data"))       // rows split over "data"
w := tensor.Shard(weight, mesh, tensor.P("", "model")) // columns over "model"
y := x.MatMul(w).ReLU()                                // each device computes its block
s, _ := y.Sharding()                                   // P("data", "model")
```

A sharded tensor is an ordinary lazy tensor whose device is the mesh, so
autograd is unchanged, and realizing it partitions the graph. Each
operation gets a layout by a few rules. Elementwise operations stay local.
A product over a split contracted axis gives partial sums, which an
all-reduce adds up, or a reduce-scatter when the result stays split; a
reduction over a split axis reduces locally and then across devices.
Movement operations carry splits along, and attention splits over batch
and heads. Where a product's operands disagree, the one split along its
rows or batch keeps its split and the other is gathered, so the
activations of data parallelism stay split and fully sharded weights are
gathered where they are used. Anything else is made whole on every device
first, which is correct but slower, so every result matches one device's.
`tensor.WithSharding` constrains a layout, like
`jax.lax.with_sharding_constraint`, `Sharding` reports one before
anything runs, `To` gathers a tensor onto one device, and `Mesh.Traffic`
counts the collectives that have run.

`Param.Shard` and `nn.Shard` lay a model's parameters out over a mesh as a
rule says, and `train.Sharded` trains such a model, splitting each batch
over the mesh axis named `data`:

```go
mesh, err := tensor.NewMesh([]int{2, 4}, []string{"data", "model"}, tensor.GPUs()...)
if err != nil {
	return err
}
defer mesh.Close()
err = train.Fit(ctx, model, opt, trainSet,
	train.Loss(loss.CrossEntropy),
	train.Sharded(mesh, nn.TensorParallel(model, mesh, "model")))
```

With no rule a mesh trains data-parallel and matches `DataParallel`.
`nn.TensorParallel` gives Megatron's layout: column- then row-parallel
MLPs and attention split by heads, one all-reduce per row-parallel layer.
`nn.FullyShard` splits every parameter along its first evenly divisible
axis, as FSDP and ZeRO-3 do, so products gather weights where they are
used and gradients and optimizer states are sharded alike. `nn.Rules`
combines rules, for data-parallel replicas of a tensor-parallel model or
FSDP with tensor parallelism. An MLP and a GPT-2 block trained each way,
and on 2 × 2 meshes, match one device.

Splits must be even: an axis's length must divide by the size of the mesh
axis that splits it. `Sharded` does not combine with `Checkpoint`,
`MicroBatch`, `MemoryBudget` or `DataParallel`. Random tensors, such as
dropout's masks, are drawn whole on every device and then cut, and there
is no all-to-all yet, so moving a split from one axis to another gathers
the tensor and cuts it again.

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
`TransformerEncoderLayer`. Self-attention and rotary embeddings cut their
features apart with `Split`, which takes parts of given sizes along an
axis like `torch.split` and joins their gradients with one copy.

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

## Distributions and generative models

Package `distributions` follows `torch.distributions`: `Normal`,
`MultivariateNormal` (from a lower-triangular scale or a covariance),
`Bernoulli` and `Categorical` (from probabilities or logits), `Uniform`,
`Exponential`, `Laplace`, `Gamma`, `Beta`, `Dirichlet` and `Poisson`. Each
samples with a caller's seed, the same on every device, and reports its
log density or mass, entropy, mean and variance. The continuous families
also draw reparameterized samples with `RSample`, whose gradients flow to
the parameters; `Gamma`, `Beta` and `Dirichlet` get theirs from the
implicit gradients of their gamma samples. `KL` returns the divergence in
closed form between two distributions of a family, and
`distributions.ErrNoKL` for other pairs:

```go
q := distributions.NewNormal(mu, logVar.Mul(tensor.Scalar[float32](0.5)).Exp())
z := q.RSample(seed) // gradients flow to mu and logVar
kl := distributions.KL(q, prior).Sum()
```

They rest on new elementwise operations with gradients, `Log1p`, `Expm1`,
`Erf`, `Erfinv`, `Lgamma` and `Digamma`, and on the samplers
`tensor.RandGamma`, whose gradient is the implicit reparameterization
gradient, and `tensor.RandPoisson`. The samplers run on the CPU, so GPU
tensors make a round trip.

Three examples train generative models on MNIST and write PNG grids of
their samples: `examples/vae` a variational autoencoder that prints its
ELBO, `examples/gan` a GAN of two perceptrons, and `examples/diffusion` a
DDPM with ancestral and DDIM samplers, which also trains on a toy cloud of
two moons:

```sh
go run ./examples/vae -epochs 10 -out vae
go run ./examples/diffusion -dataset moons -out moons
```

## Graph networks

`tensor.Pattern` is a sparse matrix of ones, such as a graph's adjacency,
built once on the host from coordinate lists. Its `MatMul`, `Mean` and
`Max` combine rows of a tensor over each row's entries with one segment
kernel on every device, without a tensor per edge, and their gradients
are products by the transpose, so they differentiate again and batch
under `VMap`. `nn.Graph` holds the edges of a directed graph, as PyTorch
Geometric's `edge_index` does, and passes messages over them: `Source` and
`Target` read node features onto edges, `Sum`, `Mean` and `Max` aggregate
them into nodes, and `Softmax` normalizes edge scores over each node's
incoming edges. `GCNConv`, `SAGEConv` and `GATConv` are built on it:

```go
g, err := nn.NewGraph(nodes, src, dst) // an edge from src[e] to dst[e]
if err != nil {
	return err
}
conv1, conv2 := nn.NewGCNConv(1433, 16), nn.NewGCNConv(16, 7)
logits := conv2.Convolve(g, conv1.Convolve(g, x).ReLU())
```

`examples/gnn` trains this network to classify the papers of a citation
graph with Cora's sizes, generated in code so nothing is downloaded.

## Reinforcement learning

`examples/rl/cartpole` simulates Gymnasium's CartPole-v1 in pure Go, and
two examples learn it until the greedy policy's mean return over 100
episodes reaches 195: `examples/rl/dqn` with a double deep Q-network and a
replay buffer, and `examples/rl/ppo` with proximal policy optimization
over eight environments side by side and generalized advantage
estimation.

```sh
go run ./examples/rl/ppo
```

## ONNX

`onnx.Open` maps an ONNX model, and `onnx.Parse` reads one from bytes.
`Inputs` and `Outputs` describe what it takes and computes, and `Run`
turns the whole graph into lazy mimir operations and realizes the outputs
together, so a model is one schedule that the cache replays from the
second run, on any device:

```go
m, err := onnx.Open("model.onnx")
if err != nil {
	return err
}
defer m.Close()
out, err := m.Run(ctx, map[string]*tensor.Tensor{"input": x})
logits := out["output"]
```

It reads opsets `onnx.MinOpset` to `onnx.MaxOpset` (11 to 22) and runs
about ninety operators: the elementwise and comparison operators, `MatMul`
and `Gemm`, reductions, the shape, joining and indexing operators, `Conv`
and `ConvTranspose` of any number of spatial axes, the poolings and
normalizations, `Softmax`, `Gelu`, `Resize`, `Trilu`, `Range` and `CumSum`.
Shapes and indices that a graph computes, such as a `Reshape`'s target, are
computed on the host, and nodes of initializers alone are folded when the
model opens. Attention that an exporter writes as products, a scale, a
mask and a softmax becomes one `ScaledDotProductAttention` kernel.
Anything else, such as `Loop`, `If`, `Scan` or a cubic `Resize`, is an
`*onnx.UnsupportedError` that names it. The reader bounds every length by
the size of the file, so a corrupt model is an error, never a panic.
`examples/onnx` runs a model on random inputs and times it:

```sh
go run ./examples/onnx -model io/onnx/testdata/transformer.onnx -dim batch=4 -dim seq=16 -intmax 50
```

## Requirements

Go 1.27 or later. The tensor API uses generic methods (`t.Data[float32](ctx)`).
mimir depends on two modules: [purego](https://github.com/ebitengine/purego)
to call Metal, CUDA, NCCL and ROCm without cgo, and
[klauspost/compress](https://github.com/klauspost/compress) for the
Snappy and zstd pages of Parquet files, which the standard library cannot
read.

## Goals

- **Fast on CPU and GPU.** One graph, one set of kernels per device, chosen at
  run time: Apple Metal, NVIDIA CUDA and AMD ROCm today, and several
  devices at once, data-parallel or sharded over a mesh.
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
| Create | `FromSlice`, `Full`, `Zeros`, `Ones`, `Scalar`, `Arange`, `Rand`, `Randn`, `RandGamma`, `RandPoisson` |
| Elementwise | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Exp`, `Exp2`, `Log`, `Log2`, `Sin`, `Cos`, `Sqrt`, `Recip`, `Abs`, `Maximum`, `Minimum`, `Less`, `Greater`, `Equal`, `Fmod`, `Where`, `Cast` |
| Special functions | `Log1p`, `Expm1`, `Erf`, `Erfinv`, `Lgamma`, `Digamma` |
| Activations | `ReLU`, `Sigmoid`, `Tanh`, `GELU` (tanh approximation), `SiLU`, `Softmax`, `LogSoftmax` |
| Reductions | `Sum`, `Max`, `Min`, `Mean`, `Var` |
| Shape | `Reshape`, `Permute`, `Transpose`, `Expand`, `Pad`, `Shrink`, `Flip`, `Unsqueeze`, `Squeeze` |
| Linear algebra | `MatMul` with NumPy semantics and batch broadcasting, `MatMulT` (also by quantized weights) |
| Indexing | `Gather`, `ScatterAdd`, `Argmax` |
| Joining and windows | `Cat`, `Stack`, `Split`, `Flatten`, `Unfold`, `SliceScatter` |
| Sparse | `Pattern` with `MatMul`, `Mean` and `Max` over its entries, as one segment kernel |
| Scans | `CumSum`, `CumMax`, `LogCumSumExp`, `Scan` |
| Attention | `ScaledDotProductAttention` (causal, masks, grouped-query) |
| Sorting | `Sort`, `Argsort`, `TopK` (stable; NaN sorts last) |
| Fourier | `FFT`, `IFFT`, `RFFT`, `IRFFT` (any length; complex numbers as a trailing axis of 2) |
| Linear algebra | `Cholesky`, `TriangularSolve`, `Solve`, `Inv`, `Det`, `SlogDet`, `QR`, `SVD`, `Eigh` (batched, with gradients; on the CPU, so GPU tensors make a round trip) |
| Convolution | `Conv2d` (stride, padding, dilation, groups), `MaxPool2d`, `AvgPool2d` |
| Autograd | `Backward`, `Grad`, `RequiresGrad`, `Detach`, `Param`, `Func`, `Checkpoint` |
| Memory | `SetMemoryBudget`, `ErrMemoryBudget`, `Donate`, `ErrDonated`, `FromBytes` over mapped files |
| Devices | `CPUs`, `GPUs`, `NewGroup` with `AllReduce`, `Broadcast`, `AllGather`, `ReduceScatter`, `ReduceGrads`, `BroadcastParams` (CPU devices, or CUDA GPUs over NCCL) |
| Sharding | `NewMesh`, `P`, `PartitionSpec` (`WithSections`), `Shard`, `WithSharding`, `Sharding`, `Param.Shard`, `Mesh.Traffic` |
| Layers (`nn`) | `Linear`, `Conv2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`, `Sequential`, `Checkpoint`, activations, pooling, `Flatten`, `Cast` |
| Sequence layers (`nn`) | `RNN`, `LSTM`, `GRU`, `MultiheadAttention`, `SelfAttention`, `RotaryEmbedding`, `KVCache`, `MLP`, `GatedMLP`, `DecoderBlock`, `TransformerEncoderLayer` |
| Graph layers (`nn`) | `Graph` (`Source`, `Target`, `Sum`, `Mean`, `Max`, `Softmax`, `Adjacency`), `GCNConv`, `SAGEConv`, `GATConv` |
| Placement (`nn`) | `Replicate`, `Shard`, `Rules`, `FullyShard`, `TensorParallel` |
| Losses (`nn/loss`) | `CrossEntropy`, `NLL`, `BCEWithLogits`, `MSE`, `L1`, `Huber` |
| Optimizers (`optim`) | `SGD` (momentum, Nesterov), `Adam`, `AdamW`, `RMSprop`, `MasterWeights`; `StepDecay`, `Cosine`, `Warmup` schedules |
| Distributions (`distributions`) | `Normal`, `MultivariateNormal`, `Bernoulli`, `Categorical`, `Uniform`, `Exponential`, `Laplace`, `Gamma`, `Beta`, `Dirichlet`, `Poisson`; `Sample`, `RSample`, `LogProb`, `Entropy`, `Mean`, `Variance`, `KL` |
| Data (`data`, `data/datasets`) | batching, shuffling, `Stream`, `Statistics`, `Normalize`; MNIST, CSV, `CSVShards`, `ParquetShards`, `Mapped`, `Synthetic` |
| Files (`io`) | safetensors read, write and map; Parquet; GGUF read (F32, F16, BF16, Q8_0, Q4_K, Q6_K) and write; ONNX read and run (opsets 11 to 22) |
| Training (`train`, `metrics`) | `Fit` with evaluation, schedules, early stopping, checkpoints and exact resume, `MicroBatch`, `MemoryBudget`, `LossScale`, `DynamicLossScale`, `DataParallel`, `Sharded`, `Parallel`; `Accuracy` |

Element types are `float32`, `int32`, `bfloat16` and `float16`, plus the
block-quantized `Q8_0`, `Q4_K` and `Q6_K` for weights. Broadcasting follows NumPy.
Modules are plain structs: `nn.Parameters` finds every `*tensor.Param` a model
holds, in exported or unexported fields, so custom layers need no
registration.

## Layout

```
tensor/              public Tensor API, autograd, device groups, meshes and sharding
nn/                  layers, parameter discovery, save and load, replicas and sharding rules
nn/loss/             loss functions
optim/               optimizers and learning-rate schedules
distributions/       probability distributions, sampling, log densities and KL divergences
data/                datasets, batching, streams of shards and their statistics
data/datasets/       MNIST, CSV, CSV and Parquet shards, mapped safetensors, synthetic rows
train/               Fit, evaluation, checkpoints, data-parallel and sharded training
metrics/             accuracy
backend/             Backend interface, DType, Op, View, Kernel, collectives
backend/cpu/         pure-Go kernels and worker pool, CPU devices on streams, collectives
backend/metal/       Metal backend: Metal Performance Shaders and tensor operations, through purego
backend/cuda/        CUDA backend: driver API, NVRTC, cuBLAS and NCCL, through purego
backend/rocm/        ROCm backend: HIP runtime, hiprtc and hipBLAS, through purego
backend/internal/gen/        generates the kernels, in Metal Shading Language, CUDA C++ or HIP C++
backend/internal/kerneltest/ kernel cases for every device, and a CPU emulator for all four dialects
backend/internal/diskcache/  the on-disk cache of compiled CUDA and HIP kernels
internal/graph/      lazy IR with shape and dtype checks
internal/sched/      runs graphs: views, buffer reuse, freeing, the schedule cache, collectives
internal/filemap/    read-only file mappings
internal/goldentest/ golden-file loading and comparison
internal/quanttest/  reference quantization for tests
io/safetensors/      safetensors reader, writer and file mapping
io/parquet/          Parquet reader: flat schemas, every standard encoding, Snappy, gzip and zstd
io/gguf/             GGUF reader and writer
io/onnx/             ONNX model reader and interpreter
examples/mnist/      trains an MLP or CNN on MNIST
examples/gpt2/       trains a GPT-2 on a text file and samples from it
examples/llama/      runs Llama-architecture GGUF models
examples/vae/        trains a variational autoencoder on MNIST
examples/gan/        trains a GAN on MNIST
examples/diffusion/  trains a diffusion model on MNIST or two moons
examples/gnn/        trains a GCN on a generated citation graph
examples/rl/cartpole/        the CartPole environment
examples/rl/dqn/     learns CartPole with a deep Q-network
examples/rl/ppo/     learns CartPole with proximal policy optimization
examples/onnx/       runs an ONNX model and times it
examples/internal/grid/      writes samples of the generative examples as PNGs
testdata/golden/     expected results from PyTorch or NumPy
tools/asmgen/        generates the CPU assembly kernels with avo
tools/devices/       lists the devices mimir can use
tools/bench/         PyTorch and NumPy counterparts of the benchmarks
tools/golden/        golden-file generator (Python, development only)
tools/parquet/       Parquet test-file generator (Python, development only)
tools/gguf/          GGUF test-file generators (Python, development only)
tools/onnx/          ONNX golden-model generator (Python, development only)
ci/cuda/             CUDA tests and benchmarks on a Modal T4 or any NVIDIA machine, and multi-GPU runs
ci/rocm/             ROCm tests and benchmarks on any AMD machine
```

## Testing

```sh
go test -race ./...
MIMIR_EAGER=1 go test ./...
MIMIR_DEVICE=gpu go test ./...   # on a Mac, an NVIDIA or an AMD machine: every test on the GPU
MIMIR_EMULATE=1 go test -timeout=100m -run Kernels ./backend/...   # the GPU kernels of four dialects on the CPU, with clang++
go test -run='^$' -fuzz=FuzzOps -fuzztime=60s ./tensor
go test -run='^$' -fuzz=FuzzOpen -fuzztime=60s -fuzzminimizetime=0 ./io/parquet
go test -run='^$' -fuzz=FuzzOpen -fuzztime=60s ./io/gguf
go test -run='^$' -fuzz=FuzzOpen -fuzztime=60s ./io/onnx
go test -run='^$' -fuzz=FuzzRun -fuzztime=60s ./io/onnx    # a node of every operator
MIMIR_SCALE_ROWS=1000000000 go test -run=FlatMemory -timeout=85m -v ./train   # the billion-row run
```

The scale test trains over a million rows by default; `MIMIR_SCALE_ROWS`
sets another number. The Parquet test files in `io/parquet/testdata` and
`data/datasets/testdata` were written by pyarrow; `python3
tools/parquet/gen.py` writes them again.

`testdata/golden/` holds inputs and expected results: every operation's
output, the gradients of operations and layers, every layer and loss, and
optimizer steps and learning-rate schedules, scans, recurrent and
transformer layers, quantized weights, factorizations, function
transforms, probability distributions and special functions, and graph
aggregations and layers. The committed files were computed
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

`io/onnx/testdata` holds four small models, a perceptron, a CNN, a
convolutional decoder and a causal transformer, with NumPy's outputs.
`python3 tools/onnx/gen.py` writes them again with a protocol buffer
writer that needs only NumPy, and with `--oracle torch` it exports the same
networks with `torch.onnx.export` instead, which CI checks mimir against:

```sh
pip install numpy torch onnx
python3 tools/onnx/gen.py --oracle torch --out /tmp/onnx
MIMIR_ONNX_DIR=/tmp/onnx go test -run TestGolden ./io/onnx
```

GitHub has Apple silicon runners but no free NVIDIA ones, so the CUDA tests
run on a [Modal](https://modal.com) T4 through `ci/cuda/modal_test.py`, which
copies the checkout into a CUDA runtime container and runs `ci/cuda/test.sh`
there. The CI job `cuda` runs it when the repository has the secrets
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, and skips with a warning
otherwise. `pip install modal && modal run ci/cuda/modal_test.py` runs it
from a laptop, and `sh ci/cuda/test.sh` runs the same tests on any Linux
machine with an NVIDIA GPU. `ci/cuda/perf.sh` runs the GPU benchmarks
there, each with CUDA graphs off and then on (`modal run
ci/cuda/modal_test.py --script ci/cuda/perf.sh` on Modal). The CUDA
kernels are also compiled with the real NVRTC on every change, which
needs no GPU.

The multi-GPU tests run on demand, on paid Modal hosts with several GPUs,
not on every change. `ci/cuda/parallel.sh` runs NCCL's collectives,
replicas and data-parallel training on every GPU against one device, and
the data-parallel scaling benchmark; `ci/cuda/shard.sh` runs sharded
training against one device and the sharded scaling benchmark, and ends
with the 8-GPU gate, which fails when examples per second on 8 GPUs fall
below 85% of 8 times those on one:

```sh
modal run ci/cuda/modal_test.py --gpus 4 --script ci/cuda/parallel.sh
modal run ci/cuda/modal_test.py --gpus 8 --gpu-type H100 --script ci/cuda/shard.sh
```

Without several GPUs, the same tests run on `tensor.CPUs` devices, and
`BenchmarkDataParallel` and `BenchmarkSharded` in `train` report examples
per second, efficiency and, for sharding, communication per step on 1 to
4 of them.

GitHub has no AMD GPU runners either, and no CI job needs one.
`sh ci/rocm/test.sh` runs every test with `MIMIR_DEVICE=rocm`, then again
with HIP graphs, on any Linux machine with an AMD GPU, ROCm 5.4 or later
and Go, and `ci/rocm/perf.sh` runs the GPU benchmarks there with HIP
graphs off and then on. Without one, the ROCm tests check the graph logic
against a fake runtime and the hipBLAS calls on the emulator, and wherever
hiprtc is installed, or `MIMIR_HIPRTC` names it, `go test -run HIPRTC
./backend/rocm` compiles every generated kernel for gfx90a, gfx940, gfx942
and gfx1100, or the architectures `MIMIR_HIPRTC_ARCHS` lists, which needs
no GPU. CI does that on every change with the hiprtc of ROCm 5.4, taken
from the `pytorch-triton-rocm` wheel.

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
| M7 Full coverage | function transforms (differentiable gradients, forward mode, vmap), sorting, FFT and linear algebra, distributions with VAE, GAN and diffusion examples, reinforcement learning (DQN, PPO), graph networks, ONNX import, ROCm through HIP, hiprtc and hipBLAS (done; ROCm pending validation on AMD hardware) |
| M8 Multiple devices | collectives over CPU devices and NCCL, data parallelism, tensor sharding modeled on `jax.sharding` with data, tensor and fully sharded parallelism (in progress: built and tested on CPU devices, awaiting multi-GPU runs and the 8-GPU gate of 85% scaling efficiency) |

## License

Apache 2.0. See [LICENSE](LICENSE).
