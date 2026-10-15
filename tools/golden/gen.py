#!/usr/bin/env python3
"""Generate golden test data for mimir's tensor operations.

Every case draws its inputs from a seeded NumPy generator, computes the
expected output with an oracle, and stores both in one safetensors file:

    <case>/in0, <case>/in1, ...  inputs, in argument order
    <case>/out                   expected output

The tensor package's TestGolden runs the same case on mimir and compares.

Oracles:
    torch  PyTorch on CPU in float32, the reference mimir aims to match.
    numpy  NumPy in float64, rounded to float32: the exact answer.

The committed files use whichever oracle generated them (recorded in the
file metadata); CI regenerates them with PyTorch and runs the tests again.
Random tensors always use the reference Philox4x32-10 below, since mimir's
generator is defined to be counter-based and identical on every device.

Usage:
    python3 tools/golden/gen.py --oracle numpy --out tensor/testdata/golden
"""

import argparse
import json
import math
import os
import struct

import numpy as np

F32, I32 = np.float32, np.int32


# ---------------------------------------------------------------------------
# Reference Philox4x32-10, matching backend/cpu/rand.go.

MASK = 0xFFFFFFFF


def philox(ctr, key):
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        if r:
            k0 = (k0 + 0x9E3779B9) & MASK
            k1 = (k1 + 0xBB67AE85) & MASK
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & MASK, (p0 >> 32) ^ c3 ^ k1, p0 & MASK
    return c0, c1, c2, c3


def philox_rand(seed, n, normal):
    key = (seed & MASK, (seed >> 32) & MASK)
    out = np.empty(n, dtype=F32)
    for i in range(n):
        w = philox((i // 4 & MASK, i // 4 >> 32, 0, 0), key)
        lane = i % 4
        if not normal:
            out[i] = (w[lane] >> 8) * 2.0**-24
            continue
        p = lane & ~1
        u1 = ((w[p] >> 8) + 1) * 2.0**-24
        u2 = (w[p + 1] >> 8) * 2.0**-24
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out[i] = r * (math.cos(theta) if lane % 2 == 0 else math.sin(theta))
    return out


# ---------------------------------------------------------------------------
# Inputs.

def uniform(rng, shape, lo, hi):
    return rng.uniform(lo, hi, size=shape).astype(F32)


def nonzero(rng, shape, lo=0.5, hi=4.0):
    return (uniform(rng, shape, lo, hi) * rng.choice([-1, 1], size=shape)).astype(F32)


def ints(rng, shape, lo, hi):
    return rng.integers(lo, hi, size=shape).astype(I32)


# ---------------------------------------------------------------------------
# NumPy reference implementations in float64.

def np_gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def np_log_softmax(x, axis):
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def np_scatter_add(x, axis, idx, src):
    out = x.copy()
    for pos in np.ndindex(*idx.shape):
        dst = list(pos)
        dst[axis] = idx[pos]
        out[tuple(dst)] += src[pos]
    return out


def np_layer_norm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def np_attention(q, k, v):
    s = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    return np_softmax(s, -1) @ v


def np_mlp(x, w1, b1, w2, b2):
    return np_log_softmax(np_gelu(x @ w1 + b1) @ w2 + b2, -1)


# ---------------------------------------------------------------------------
# Cases: name -> (inputs, numpy oracle, torch oracle). The torch oracle may be
# None when the operation is defined by mimir itself (random numbers).


def cases(rng):
    c = {}

    def case(name, inputs, np_fn, torch_fn):
        c[name] = (inputs, np_fn, torch_fn)

    def unary(name, lo, hi, np_fn, torch_fn):
        case(name, [uniform(rng, (4, 33), lo, hi)], np_fn, torch_fn)

    unary("neg", -4, 4, lambda x: -x, lambda T, x: -x)
    unary("exp", -10, 10, np.exp, lambda T, x: T.exp(x))
    unary("exp2", -10, 10, np.exp2, lambda T, x: T.exp2(x))
    unary("log", 0.01, 100, np.log, lambda T, x: T.log(x))
    unary("log2", 0.01, 100, np.log2, lambda T, x: T.log2(x))
    unary("sin", -6, 6, np.sin, lambda T, x: T.sin(x))
    unary("cos", -6, 6, np.cos, lambda T, x: T.cos(x))
    unary("sqrt", 0, 100, np.sqrt, lambda T, x: T.sqrt(x))
    case("recip", [nonzero(rng, (4, 33))], lambda x: 1 / x, lambda T, x: T.reciprocal(x))
    unary("abs", -4, 4, np.abs, lambda T, x: T.abs(x))
    unary("relu", -4, 4, lambda x: np.maximum(x, 0), lambda T, x: T.relu(x))
    unary("sigmoid", -20, 20, np_sigmoid, lambda T, x: T.sigmoid(x))
    unary("tanh", -6, 6, np.tanh, lambda T, x: T.tanh(x))
    unary("gelu", -6, 6, np_gelu, lambda T, x: T.nn.functional.gelu(x, approximate="tanh"))
    unary("silu", -8, 8, lambda x: x * np_sigmoid(x), lambda T, x: T.nn.functional.silu(x))

    def binary(name, np_fn, torch_fn, y=None):
        x = uniform(rng, (3, 1, 5), -4, 4)
        case(name, [x, uniform(rng, (4, 5), -4, 4) if y is None else y], np_fn, torch_fn)

    binary("add", np.add, lambda T, x, y: x + y)
    binary("sub", np.subtract, lambda T, x, y: x - y)
    binary("mul", np.multiply, lambda T, x, y: x * y)
    binary("div", np.divide, lambda T, x, y: x / y, nonzero(rng, (4, 5)))
    binary("maximum", np.maximum, lambda T, x, y: T.maximum(x, y))
    binary("minimum", np.minimum, lambda T, x, y: T.minimum(x, y))
    binary("less", lambda x, y: (x < y).astype(F32), lambda T, x, y: (x < y).float())
    binary("greater", lambda x, y: (x > y).astype(F32), lambda T, x, y: (x > y).float())
    rounded = lambda shape: np.round(uniform(rng, shape, -2, 2)).astype(F32)  # noqa: E731
    case("equal", [rounded((3, 1, 5)), rounded((4, 5))],
         lambda x, y: (x == y).astype(F32), lambda T, x, y: (x == y).float())
    binary("fmod", np.fmod, lambda T, x, y: T.fmod(x, y), nonzero(rng, (4, 5)))
    case("where", [rounded((3, 1, 5)), uniform(rng, (4, 5), -4, 4), uniform(rng, (3, 4, 1), -4, 4)],
         lambda c, x, y: np.where(c != 0, x, y), lambda T, c, x, y: T.where(c != 0, x, y))

    case("add_int", [ints(rng, (3, 4), -1000, 1000), ints(rng, (4,), -1000, 1000)],
         np.add, lambda T, x, y: x + y)
    case("mul_int", [ints(rng, (3, 4), -1000, 1000), ints(rng, (3, 1), -1000, 1000)],
         np.multiply, lambda T, x, y: x * y)
    ys = ints(rng, (3, 4), 1, 50) * rng.choice([-1, 1], size=(3, 4)).astype(I32)
    case("fmod_int", [ints(rng, (3, 4), -1000, 1000), ys], np.fmod, lambda T, x, y: T.fmod(x, y))
    case("sum_int", [ints(rng, (3, 4, 5), -1000, 1000)],
         lambda x: x.sum(axis=1, dtype=I32), lambda T, x: x.sum(dim=1, dtype=T.int32))
    case("cast_f2i", [uniform(rng, (4, 9), -100, 100)],
         lambda x: x.astype(I32), lambda T, x: x.to(T.int32))
    case("cast_i2f", [ints(rng, (4, 9), -100000, 100000)],
         lambda x: x.astype(F32), lambda T, x: x.to(T.float32))

    x345 = lambda: uniform(rng, (3, 4, 5), -4, 4)  # noqa: E731
    case("sum_all", [x345()], lambda x: x.sum(), lambda T, x: x.sum())
    case("sum_axis1", [x345()], lambda x: x.sum(axis=1), lambda T, x: x.sum(dim=1))
    case("sum_axes02", [x345()], lambda x: x.sum(axis=(0, 2)), lambda T, x: x.sum(dim=(0, 2)))
    case("max_axis2", [x345()], lambda x: x.max(axis=2), lambda T, x: x.amax(dim=2))
    case("min_axis0", [x345()], lambda x: x.min(axis=0), lambda T, x: x.amin(dim=0))
    case("mean_last", [x345()], lambda x: x.mean(axis=-1), lambda T, x: x.mean(dim=-1))
    case("var_axis1", [x345()], lambda x: x.var(axis=1, ddof=1), lambda T, x: x.var(dim=1))
    case("softmax_axis1", [x345()], lambda x: np_softmax(x, 1), lambda T, x: T.softmax(x, dim=1))
    case("log_softmax_large", [uniform(rng, (3, 4, 5), -500, 500)],
         lambda x: np_log_softmax(x, -1), lambda T, x: T.log_softmax(x, dim=-1))

    case("matmul_2d", [uniform(rng, (7, 9), -1, 1), uniform(rng, (9, 5), -1, 1)],
         np.matmul, lambda T, x, y: x @ y)
    case("matmul_batched", [uniform(rng, (2, 1, 4, 6), -1, 1), uniform(rng, (3, 6, 5), -1, 1)],
         np.matmul, lambda T, x, y: x @ y)
    case("matmul_vec_mat", [uniform(rng, (6,), -1, 1), uniform(rng, (6, 4), -1, 1)],
         np.matmul, lambda T, x, y: x @ y)
    case("matmul_mat_vec", [uniform(rng, (4, 6), -1, 1), uniform(rng, (6,), -1, 1)],
         np.matmul, lambda T, x, y: x @ y)
    case("matmul_transposed", [uniform(rng, (5, 3), -1, 1), uniform(rng, (5, 4), -1, 1)],
         lambda x, y: x.T @ y, lambda T, x, y: x.T @ y)

    case("permute_reshape", [uniform(rng, (2, 3, 4), -4, 4)],
         lambda x: x.transpose(2, 0, 1).reshape(4, 6), lambda T, x: x.permute(2, 0, 1).reshape(4, 6))
    case("expand", [uniform(rng, (3, 1), -4, 4)],
         lambda x: np.broadcast_to(x, (2, 3, 4)), lambda T, x: x.expand(2, 3, 4))
    case("pad", [uniform(rng, (2, 3), -4, 4)],
         lambda x: np.pad(x, ((1, 0), (2, 1)), constant_values=-1),
         lambda T, x: T.nn.functional.pad(x, (2, 1, 1, 0), value=-1))
    case("shrink", [uniform(rng, (5, 6), -4, 4)], lambda x: x[1:4, 2:6], lambda T, x: x[1:4, 2:6])
    case("flip", [uniform(rng, (3, 4), -4, 4)], lambda x: x[::-1, ::-1], lambda T, x: T.flip(x, (0, 1)))
    case("shrink_flip_add", [uniform(rng, (5, 6), -4, 4)],
         lambda x: x[1:4, 2:6][::-1] + x[0:3, 0:4],
         lambda T, x: T.flip(x[1:4, 2:6], (0,)) + x[0:3, 0:4])

    case("gather", [uniform(rng, (3, 5), -4, 4), ints(rng, (3, 4), 0, 5)],
         lambda x, i: np.take_along_axis(x, i.astype(np.int64), axis=1),
         lambda T, x, i: T.gather(x, 1, i.long()))
    case("scatter_add", [uniform(rng, (3, 5), -4, 4), ints(rng, (2, 5), 0, 3), uniform(rng, (2, 5), -4, 4)],
         lambda x, i, s: np_scatter_add(x, 0, i, s),
         lambda T, x, i, s: x.scatter_add(0, i.long(), s))

    case("rand_uniform", [], lambda: philox_rand(7, 1001, False), None)
    case("rand_normal", [], lambda: philox_rand(7, 1001, True), None)

    case("layer_norm", [uniform(rng, (4, 16), -4, 4), uniform(rng, (16,), 0.5, 1.5), uniform(rng, (16,), -1, 1)],
         np_layer_norm, lambda T, x, g, b: T.nn.functional.layer_norm(x, (16,), g, b, eps=1e-5))
    case("attention", [uniform(rng, (2, 4, 8), -1, 1) for _ in range(3)],
         np_attention, lambda T, q, k, v: T.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), dim=-1) @ v)
    case("mlp", [uniform(rng, (8, 16), -1, 1), uniform(rng, (16, 32), -0.5, 0.5), uniform(rng, (32,), -0.1, 0.1),
                 uniform(rng, (32, 10), -0.5, 0.5), uniform(rng, (10,), -0.1, 0.1)],
         np_mlp,
         lambda T, x, w1, b1, w2, b2: T.log_softmax(
             T.nn.functional.gelu(x @ w1 + b1, approximate="tanh") @ w2 + b2, dim=-1))

    return c


def evaluate(oracle, inputs, np_fn, torch_fn):
    if oracle == "torch" and torch_fn is not None:
        import torch

        out = torch_fn(torch, *[torch.from_numpy(a) for a in inputs])
        return out.numpy()
    if not inputs:
        return np_fn()
    widened = [a.astype(np.float64) if a.dtype == F32 else a for a in inputs]
    out = np.asarray(np_fn(*widened))
    return out.astype(F32) if out.dtype == np.float64 else out


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name in sorted(tensors):
        a = np.asarray(tensors[name], order="C")  # ascontiguousarray would make 0-d arrays 1-d
        dtype = {np.dtype(F32): "F32", np.dtype(I32): "I32"}[a.dtype]
        data = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    h += b" " * (-len(h) % 8)  # align the data section to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for data in blobs:
            f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out", default=os.path.join("tensor", "testdata", "golden"))
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()

    metadata = {"oracle": args.oracle, "numpy": np.__version__, "seed": str(args.seed)}
    if args.oracle == "torch":
        import torch

        torch.set_num_threads(1)
        metadata["torch"] = torch.__version__

    tensors = {}
    for name, (inputs, np_fn, torch_fn) in cases(np.random.default_rng(args.seed)).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        out = evaluate(args.oracle, inputs, np_fn, torch_fn)
        if out.dtype not in (np.dtype(F32), np.dtype(I32)):
            raise SystemExit(f"{name}: oracle returned {out.dtype}")
        tensors[f"{name}/out"] = out

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ops.safetensors")
    write_safetensors(path, tensors, metadata)
    print(f"wrote {len(tensors)} tensors to {path} (oracle {args.oracle})")


if __name__ == "__main__":
    main()
