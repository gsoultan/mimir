#!/usr/bin/env python3
"""Generate golden test data for mimir.

Every case draws its inputs from a seeded NumPy generator, computes the
expected results with an oracle, and stores both in safetensors files:

    <case>/in0, <case>/in1, ...  inputs, in argument order
    <case>/out                   expected output
    <case>/gout                  gradient of a loss with respect to out
    <case>/grad<i>               expected gradient with respect to input i
    <case>/<name>                any other expected result, such as a
                                 layer's running statistics

Files:
    ops.safetensors     tensor operations (package tensor)
    grads.safetensors   gradients of tensor operations (package tensor)
    nn.safetensors      layers and losses with their gradients (packages nn, nn/loss)
    optim.safetensors   optimizer steps and learning-rate schedules (package optim)
    scan.safetensors    running sums, maxima and log-sum-exps (package tensor)
    rnn.safetensors     recurrent layers with their gradients (package nn)
    quant.safetensors   block-quantized weights as ggml lays them out, U8
                        blocks, dequantized and multiplied (package tensor)
    attention.safetensors  attention, transformer layers and blocks with their
                        gradients (package nn)
    linalg.safetensors  sorts, top-k, Fourier transforms and linear algebra with
                        their gradients (package tensor)

Oracles:
    torch  PyTorch on CPU, the reference mimir aims to match: float32 for
           forward results, float64 for gradients and optimizer steps.
    numpy  NumPy in float64, rounded to float32: the exact answer.
           Gradients are central differences in float64.

The committed files use whichever oracle generated them (recorded in the
file metadata); CI regenerates them with PyTorch and runs the tests again.
Random tensors always use the reference Philox4x32-10 below, since mimir's
generator is defined to be counter-based and identical on every device.

Usage:
    python3 tools/golden/gen.py --oracle numpy --out testdata/golden
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


def np_unfold(x, axis, size, step):
    w = np.lib.stride_tricks.sliding_window_view(x, size, axis=axis)
    keep = [slice(None)] * w.ndim
    keep[axis] = slice(None, None, step)
    return w[tuple(keep)]


def np_windows(x, k, stride, padding, dilation, fill):
    """[N, C, H, W] -> [N, C, OH, OW, k, k] windows, padded with fill."""
    p = padding
    x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=fill)
    span = dilation * (k - 1) + 1
    w = np_unfold(np_unfold(x, 2, span, stride), 3, span, stride)
    return w[..., ::dilation, ::dilation]


def np_conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    cout, cg, k, _ = w.shape
    win = np_windows(x, k, stride, padding, dilation, 0)  # [N, C, OH, OW, k, k]
    og = cout // groups
    parts = []
    for g in range(groups):
        xs = win[:, g * cg:(g + 1) * cg]
        parts.append(np.einsum("nchwij,ocij->nohw", xs, w[g * og:(g + 1) * og]))
    out = np.concatenate(parts, axis=1)
    return out if b is None else out + b[None, :, None, None]


def np_max_pool2d(x, k, stride=None, padding=0):
    return np_windows(x, k, stride or k, padding, 1, -np.inf).max(axis=(-2, -1))


def np_avg_pool2d(x, k, stride=None, padding=0):
    return np_windows(x, k, stride or k, padding, 1, 0).mean(axis=(-2, -1))


def np_batch_norm(x, w, b, eps=1e-5):
    axes = (0,) + tuple(range(2, x.ndim))
    shape = [1, -1] + [1] * (x.ndim - 2)
    m = x.mean(axis=axes, keepdims=True)
    v = x.var(axis=axes, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w.reshape(shape) + b.reshape(shape)


def np_cross_entropy(logits, target):
    lp = np_log_softmax(logits, 1)
    picked = np.take_along_axis(lp, target[:, None].astype(np.int64), axis=1)
    return -picked.mean()


def np_nll(logp, target):
    return -np.take_along_axis(logp, target[:, None].astype(np.int64), axis=1).mean()


def np_bce_with_logits(x, y):
    return (np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))).mean()


def np_huber(x, y, delta=1.0):
    d = np.abs(x - y)
    return np.where(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta)).mean()


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


# ---------------------------------------------------------------------------
# Gradient cases: name -> (inputs, differentiable input indices, numpy
# oracle, torch oracle). The loss is sum(out * gout).


def grad_cases(rng):
    c = {}

    def case(name, inputs, diff, np_fn, torch_fn):
        c[name] = (inputs, diff, np_fn, torch_fn)

    def unary(name, lo, hi, np_fn, torch_fn):
        case(name, [uniform(rng, (3, 5), lo, hi)], [0], np_fn, torch_fn)

    unary("exp", -2, 2, np.exp, lambda T, x: T.exp(x))
    unary("log", 0.5, 3, np.log, lambda T, x: T.log(x))
    unary("sin", -3, 3, np.sin, lambda T, x: T.sin(x))
    unary("sqrt", 0.5, 3, np.sqrt, lambda T, x: T.sqrt(x))
    case("recip", [nonzero(rng, (3, 5))], [0], lambda x: 1 / x, lambda T, x: T.reciprocal(x))
    unary("sigmoid", -4, 4, np_sigmoid, lambda T, x: T.sigmoid(x))
    unary("tanh", -2, 2, np.tanh, lambda T, x: T.tanh(x))
    unary("gelu", -3, 3, np_gelu, lambda T, x: T.nn.functional.gelu(x, approximate="tanh"))
    unary("silu", -3, 3, lambda x: x * np_sigmoid(x), lambda T, x: T.nn.functional.silu(x))
    unary("relu", -2, 2, lambda x: np.maximum(x, 0), lambda T, x: T.relu(x))
    unary("abs", -2, 2, np.abs, lambda T, x: T.abs(x))

    def binary(name, np_fn, torch_fn, y=None):
        x = uniform(rng, (3, 1, 5), -2, 2)
        case(name, [x, uniform(rng, (4, 5), -2, 2) if y is None else y], [0, 1], np_fn, torch_fn)

    binary("add", np.add, lambda T, x, y: x + y)
    binary("sub", np.subtract, lambda T, x, y: x - y)
    binary("mul", np.multiply, lambda T, x, y: x * y)
    binary("div", np.divide, lambda T, x, y: x / y, nonzero(rng, (4, 5)))
    binary("maximum", np.maximum, lambda T, x, y: T.maximum(x, y))
    binary("minimum", np.minimum, lambda T, x, y: T.minimum(x, y))
    rounded = np.round(uniform(rng, (3, 1, 5), -2, 2)).astype(F32)
    case("where", [rounded, uniform(rng, (4, 5), -2, 2), uniform(rng, (3, 4, 1), -2, 2)], [1, 2],
         lambda c, x, y: np.where(c != 0, x, y), lambda T, c, x, y: T.where(c != 0, x, y))

    x345 = lambda: uniform(rng, (3, 4, 5), -2, 2)  # noqa: E731
    case("sum_axes02", [x345()], [0], lambda x: x.sum(axis=(0, 2)), lambda T, x: x.sum(dim=(0, 2)))
    case("max_axis2", [x345()], [0], lambda x: x.max(axis=2), lambda T, x: x.amax(dim=2))
    case("mean_last", [x345()], [0], lambda x: x.mean(axis=-1), lambda T, x: x.mean(dim=-1))
    case("var_axis1", [x345()], [0], lambda x: x.var(axis=1, ddof=1), lambda T, x: x.var(dim=1))
    case("softmax_axis1", [x345()], [0], lambda x: np_softmax(x, 1), lambda T, x: T.softmax(x, dim=1))
    case("log_softmax", [x345()], [0], lambda x: np_log_softmax(x, -1), lambda T, x: T.log_softmax(x, dim=-1))

    case("permute_reshape", [uniform(rng, (2, 3, 4), -2, 2)], [0],
         lambda x: x.transpose(2, 0, 1).reshape(4, 6), lambda T, x: x.permute(2, 0, 1).reshape(4, 6))
    case("expand", [uniform(rng, (3, 1), -2, 2)], [0],
         lambda x: np.broadcast_to(x, (2, 3, 4)), lambda T, x: x.expand(2, 3, 4))
    case("pad", [uniform(rng, (2, 3), -2, 2)], [0],
         lambda x: np.pad(x, ((1, 0), (2, 1)), constant_values=-1),
         lambda T, x: T.nn.functional.pad(x, (2, 1, 1, 0), value=-1))
    case("shrink_flip_add", [uniform(rng, (5, 6), -2, 2)], [0],
         lambda x: x[1:4, 2:6][::-1] + x[0:3, 0:4],
         lambda T, x: T.flip(x[1:4, 2:6], (0,)) + x[0:3, 0:4])
    case("unfold_overlap", [uniform(rng, (2, 9, 3), -2, 2)], [0],
         lambda x: np_unfold(x, 1, 4, 2), lambda T, x: x.unfold(1, 4, 2))
    case("unfold_gapped", [uniform(rng, (2, 10), -2, 2)], [0],
         lambda x: np_unfold(x, 1, 2, 3), lambda T, x: x.unfold(1, 2, 3))
    case("cat", [uniform(rng, (2, 3), -2, 2), uniform(rng, (2, 2), -2, 2)], [0, 1],
         lambda x, y: np.concatenate([x, y], axis=1), lambda T, x, y: T.cat([x, y], dim=1))

    case("matmul_batched", [uniform(rng, (2, 1, 4, 6), -1, 1), uniform(rng, (3, 6, 5), -1, 1)], [0, 1],
         np.matmul, lambda T, x, y: x @ y)
    case("matmul_vec_mat", [uniform(rng, (6,), -1, 1), uniform(rng, (6, 4), -1, 1)], [0, 1],
         np.matmul, lambda T, x, y: x @ y)
    case("gather", [uniform(rng, (3, 5), -2, 2), ints(rng, (3, 4), 0, 5)], [0],
         lambda x, i: np.take_along_axis(x, i.astype(np.int64), axis=1),
         lambda T, x, i: T.gather(x, 1, i.long()))
    case("scatter_add", [uniform(rng, (3, 5), -2, 2), ints(rng, (2, 5), 0, 3), uniform(rng, (2, 5), -2, 2)], [0, 2],
         lambda x, i, s: np_scatter_add(x, 0, i, s),
         lambda T, x, i, s: x.scatter_add(0, i.long(), s))

    case("conv2d", [uniform(rng, (2, 3, 6, 6), -1, 1), uniform(rng, (4, 3, 3, 3), -1, 1), uniform(rng, (4,), -1, 1)],
         [0, 1, 2], lambda x, w, b: np_conv2d(x, w, b, padding=1),
         lambda T, x, w, b: T.nn.functional.conv2d(x, w, b, padding=1))
    case("conv2d_grouped", [uniform(rng, (1, 4, 9, 9), -1, 1), uniform(rng, (6, 2, 3, 3), -1, 1)], [0, 1],
         lambda x, w: np_conv2d(x, w, stride=2, padding=1, dilation=2, groups=2),
         lambda T, x, w: T.nn.functional.conv2d(x, w, stride=2, padding=1, dilation=2, groups=2))
    case("max_pool2d", [uniform(rng, (2, 3, 6, 7), -2, 2)], [0],
         lambda x: np_max_pool2d(x, 2), lambda T, x: T.nn.functional.max_pool2d(x, 2))
    case("max_pool2d_padded", [uniform(rng, (1, 2, 7, 7), -2, 2)], [0],
         lambda x: np_max_pool2d(x, 3, 2, 1), lambda T, x: T.nn.functional.max_pool2d(x, 3, 2, 1))
    case("avg_pool2d", [uniform(rng, (1, 2, 6, 6), -2, 2)], [0],
         lambda x: np_avg_pool2d(x, 3, 1, 1), lambda T, x: T.nn.functional.avg_pool2d(x, 3, 1, 1))

    case("layer_norm", [uniform(rng, (4, 16), -2, 2), uniform(rng, (16,), 0.5, 1.5), uniform(rng, (16,), -1, 1)],
         [0, 1, 2], np_layer_norm, lambda T, x, g, b: T.nn.functional.layer_norm(x, (16,), g, b, eps=1e-5))
    case("attention", [uniform(rng, (2, 4, 8), -1, 1) for _ in range(3)], [0, 1, 2],
         np_attention, lambda T, q, k, v: T.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), dim=-1) @ v)
    case("mlp", [uniform(rng, (8, 16), -1, 1), uniform(rng, (16, 32), -0.5, 0.5), uniform(rng, (32,), -0.1, 0.1),
                 uniform(rng, (32, 10), -0.5, 0.5), uniform(rng, (10,), -0.1, 0.1)], [0, 1, 2, 3, 4],
         np_mlp,
         lambda T, x, w1, b1, w2, b2: T.log_softmax(
             T.nn.functional.gelu(x @ w1 + b1, approximate="tanh") @ w2 + b2, dim=-1))
    return c


def nn_cases(rng):
    """Layers and losses, as gradient cases. Extra results are returned by
    the oracles as a dict next to the output."""
    c = {}

    def case(name, inputs, diff, np_fn, torch_fn):
        c[name] = (inputs, diff, np_fn, torch_fn)

    case("linear", [uniform(rng, (4, 6), -1, 1), uniform(rng, (3, 6), -1, 1), uniform(rng, (3,), -1, 1)], [0, 1, 2],
         lambda x, w, b: x @ w.T + b, lambda T, x, w, b: T.nn.functional.linear(x, w, b))
    case("layer_norm", [uniform(rng, (3, 4, 8), -2, 2), uniform(rng, (8,), 0.5, 1.5), uniform(rng, (8,), -1, 1)],
         [0, 1, 2], np_layer_norm, lambda T, x, w, b: T.nn.functional.layer_norm(x, (8,), w, b, eps=1e-5))

    def batch_norm(name, shape):
        ch = shape[1]

        def np_fn(x, w, b):
            axes = (0,) + tuple(range(2, x.ndim))
            n = x.size // ch
            mean, var = x.mean(axis=axes), x.var(axis=axes)
            extra = {"running_mean": 0.1 * mean, "running_var": 0.9 + 0.1 * var * n / (n - 1)}
            return np_batch_norm(x, w, b), extra

        def torch_fn(T, x, w, b):
            rm, rv = T.zeros(ch, dtype=x.dtype), T.ones(ch, dtype=x.dtype)
            out = T.nn.functional.batch_norm(x, rm, rv, w, b, training=True, momentum=0.1, eps=1e-5)
            return out, {"running_mean": rm, "running_var": rv}

        case(name, [uniform(rng, shape, -2, 2), uniform(rng, (ch,), 0.5, 1.5), uniform(rng, (ch,), -1, 1)],
             [0, 1, 2], np_fn, torch_fn)

    batch_norm("batch_norm2d", (4, 3, 2, 3))
    batch_norm("batch_norm1d", (8, 5))
    case("batch_norm_eval",
         [uniform(rng, (4, 3, 2, 2), -2, 2), uniform(rng, (3,), -0.5, 0.5), uniform(rng, (3,), 0.5, 2),
          uniform(rng, (3,), 0.5, 1.5), uniform(rng, (3,), -1, 1)], [0, 3, 4],
         lambda x, rm, rv, w, b: (x - rm[:, None, None]) / np.sqrt(rv[:, None, None] + 1e-5) * w[:, None, None]
         + b[:, None, None],
         lambda T, x, rm, rv, w, b: T.nn.functional.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5))
    case("embedding", [ints(rng, (2, 3), 0, 5), uniform(rng, (5, 4), -1, 1)], [1],
         lambda i, w: w[i.astype(np.int64)], lambda T, i, w: T.nn.functional.embedding(i.long(), w))

    case("cross_entropy", [uniform(rng, (6, 5), -3, 3), ints(rng, (6,), 0, 5)], [0],
         np_cross_entropy, lambda T, x, t: T.nn.functional.cross_entropy(x, t.long()))
    case("cross_entropy_kdim", [uniform(rng, (2, 4, 3), -3, 3), ints(rng, (2, 3), 0, 4)], [0],
         lambda x, t: np_cross_entropy(np.moveaxis(x, 1, -1).reshape(-1, 4), t.reshape(-1)),
         lambda T, x, t: T.nn.functional.cross_entropy(x, t.long()))
    case("nll", [uniform(rng, (6, 5), -3, 0), ints(rng, (6,), 0, 5)], [0],
         np_nll, lambda T, x, t: T.nn.functional.nll_loss(x, t.long()))
    case("bce_with_logits", [uniform(rng, (3, 4), -3, 3), uniform(rng, (3, 4), 0, 1)], [0, 1],
         np_bce_with_logits, lambda T, x, y: T.nn.functional.binary_cross_entropy_with_logits(x, y))
    case("mse", [uniform(rng, (3, 4), -2, 2), uniform(rng, (3, 4), -2, 2)], [0, 1],
         lambda x, y: ((x - y) ** 2).mean(), lambda T, x, y: T.nn.functional.mse_loss(x, y))
    case("l1", [uniform(rng, (3, 4), -2, 2), uniform(rng, (3, 4), -2, 2)], [0, 1],
         lambda x, y: np.abs(x - y).mean(), lambda T, x, y: T.nn.functional.l1_loss(x, y))
    case("huber", [uniform(rng, (3, 4), -2, 2), uniform(rng, (3, 4), -2, 2)], [0, 1],
         np_huber, lambda T, x, y: T.nn.functional.huber_loss(x, y, delta=1.0))
    return c


def np_cummax_indices(x, axis):
    """torch.cummax's indices: a tie or a NaN takes the latest position."""
    x = np.moveaxis(x, axis, -1)
    idx = np.zeros(x.shape, dtype=I32)
    best, at = x[..., 0], np.zeros(x.shape[:-1], dtype=I32)
    for j in range(x.shape[-1]):
        take = np.isnan(x[..., j]) | (~np.isnan(best) & (x[..., j] >= best))
        best, at = np.where(take, x[..., j], best), np.where(take, j, at).astype(I32)
        idx[..., j] = at
    return np.moveaxis(idx, -1, axis)


def scan_cases(rng):
    """Scans, as gradient cases. A case without differentiable inputs checks
    its output alone."""
    c = {}

    def case(name, inputs, diff, np_fn, torch_fn):
        c[name] = (inputs, diff, np_fn, torch_fn)

    case("cumsum_axis1", [uniform(rng, (3, 4, 5), -2, 2)], [0],
         lambda x: np.cumsum(x, axis=1), lambda T, x: T.cumsum(x, dim=1))
    case("cumsum_last", [uniform(rng, (4, 33), -2, 2)], [0],
         lambda x: np.cumsum(x, axis=-1), lambda T, x: T.cumsum(x, dim=-1))
    case("cumsum_int", [ints(rng, (3, 7), -1000, 1000)], [],
         lambda x: np.cumsum(x, axis=1, dtype=I32), lambda T, x: T.cumsum(x, dim=1, dtype=T.int32))
    case("cummax_axis1", [uniform(rng, (3, 6, 4), -2, 2)], [0],
         lambda x: np.maximum.accumulate(x, axis=1), lambda T, x: T.cummax(x, dim=1).values)
    case("cummax_axis0", [uniform(rng, (7, 3), -2, 2)], [0],
         lambda x: np.maximum.accumulate(x, axis=0), lambda T, x: T.cummax(x, dim=0).values)
    ties = np.round(uniform(rng, (4, 9), -2, 2)).astype(F32)
    case("cummax_indices", [ties], [],
         lambda x: np_cummax_indices(x, 1), lambda T, x: T.cummax(x, dim=1).indices.to(T.int32))
    case("logcumsumexp_last", [uniform(rng, (3, 7), -5, 5)], [0],
         lambda x: np.logaddexp.accumulate(x, axis=-1), lambda T, x: T.logcumsumexp(x, dim=-1))
    case("logcumsumexp_large", [uniform(rng, (6, 4), -500, 500)], [0],
         lambda x: np.logaddexp.accumulate(x, axis=0), lambda T, x: T.logcumsumexp(x, dim=0))
    return c


# ---------------------------------------------------------------------------
# Optimizers minimize sum(scale * (p - target)^2) for STEPS steps from p0;
# out is the final p. Schedules record the learning rate before each step.

STEPS = 5


def np_optimize(kind, p, t, s, **kw):
    p, t, s = (a.astype(np.float64) for a in (p, t, s))
    lr = kw["lr"]
    state = {}
    for step in range(1, STEPS + 1):
        g = 2 * s * (p - t)
        if kind == "sgd":
            wd, mom, nest = kw.get("weight_decay", 0), kw.get("momentum", 0), kw.get("nesterov", False)
            g = g + wd * p
            if mom:
                buf = state["buf"] = g if step == 1 else mom * state["buf"] + g
                g = g + mom * buf if nest else buf
            p = p - lr * g
        elif kind in ("adam", "adamw"):
            b1, b2, eps, wd = 0.9, 0.999, 1e-8, kw.get("weight_decay", 0)
            if kind == "adamw":
                p = p * (1 - lr * wd)
            else:
                g = g + wd * p
            m = state["m"] = b1 * state.get("m", 0) + (1 - b1) * g
            v = state["v"] = b2 * state.get("v", 0) + (1 - b2) * g * g
            p = p - lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + eps)
        elif kind == "rmsprop":
            alpha, eps = kw.get("alpha", 0.99), 1e-8
            v = state["v"] = alpha * state.get("v", 0) + (1 - alpha) * g * g
            p = p - lr * g / (np.sqrt(v) + eps)
    return p


def torch_optimize(T, kind, p, t, s, **kw):
    p = T.tensor(p, dtype=T.float64, requires_grad=True)
    t, s = T.tensor(t, dtype=T.float64), T.tensor(s, dtype=T.float64)
    cls = {"sgd": T.optim.SGD, "adam": T.optim.Adam, "adamw": T.optim.AdamW, "rmsprop": T.optim.RMSprop}[kind]
    if kind == "adamw":
        kw.setdefault("weight_decay", 0)
    opt = cls([p], **kw)
    for _ in range(STEPS):
        opt.zero_grad()
        (s * (p - t) ** 2).sum().backward()
        opt.step()
    return p.detach().numpy()


OPTIMIZERS = {
    "sgd": ("sgd", {"lr": 0.05}),
    "sgd_momentum": ("sgd", {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.01}),
    "sgd_nesterov": ("sgd", {"lr": 0.05, "momentum": 0.9, "nesterov": True}),
    "adam": ("adam", {"lr": 0.1}),
    "adam_l2": ("adam", {"lr": 0.1, "weight_decay": 0.1}),
    "adamw": ("adamw", {"lr": 0.1, "weight_decay": 0.1}),
    "rmsprop": ("rmsprop", {"lr": 0.01, "alpha": 0.9}),
}

SCHEDULES = {
    # name: (numpy lr(epoch), torch scheduler factory, epochs)
    "step": (lambda e: 0.1 * 0.5 ** (e // 2), lambda T, o: T.optim.lr_scheduler.StepLR(o, 2, 0.5), 7),
    "cosine": (lambda e: 0.01 + (0.1 - 0.01) * (1 + math.cos(math.pi * e / 5)) / 2,
               lambda T, o: T.optim.lr_scheduler.CosineAnnealingLR(o, 5, 0.01), 6),
    "warmup": (lambda e: 0.1 * (0.25 + 0.75 * min(e, 4) / 4),
               lambda T, o: T.optim.lr_scheduler.LinearLR(o, 0.25, 1.0, 4), 7),
}


def optim_tensors(oracle, rng):
    tensors = {}
    p0, t, s = uniform(rng, (8,), -2, 2), uniform(rng, (8,), -2, 2), uniform(rng, (8,), 0.5, 2)
    for name, (kind, kw) in OPTIMIZERS.items():
        tensors.update({f"{name}/in0": p0, f"{name}/in1": t, f"{name}/in2": s})
        if oracle == "torch":
            import torch

            out = torch_optimize(torch, kind, p0, t, s, **dict(kw))
        else:
            out = np_optimize(kind, p0, t, s, **kw)
        tensors[f"{name}/out"] = out.astype(F32)
    for name, (np_lr, torch_sched, epochs) in SCHEDULES.items():
        if oracle == "torch":
            import torch

            opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
            sched, lrs = torch_sched(torch, opt), []
            for _ in range(epochs):
                lrs.append(opt.param_groups[0]["lr"])
                opt.step()
                sched.step()
        else:
            lrs = [np_lr(e) for e in range(epochs)]
        tensors[f"{name}/out"] = np.array(lrs, dtype=F32)
    return tensors


# ---------------------------------------------------------------------------
# Recurrent layers (rnn.safetensors). The inputs of a case are x, the
# initial states (h0, and c0 for an LSTM) and the parameters in the order
# of torch's named_parameters. out joins the flattened output, h_n and, for
# an LSTM, c_n, so the gradients cover all three; they are also stored on
# their own as output, h_n and c_n.

RNN_CASES = {
    # name: (kind, input size, hidden size, layers, bidirectional, batch_first, relu, bias, batched, T, B)
    "rnn_tanh": ("rnn", 3, 4, 1, False, False, False, True, True, 4, 2),
    "rnn_relu_2layer_bidir": ("rnn", 3, 4, 2, True, False, True, True, True, 4, 2),
    "lstm": ("lstm", 3, 4, 1, False, False, False, True, True, 4, 2),
    "lstm_2layer_bidir": ("lstm", 3, 4, 2, True, False, False, True, True, 4, 2),
    "lstm_batch_first_unbiased": ("lstm", 3, 4, 1, False, True, False, False, True, 5, 2),
    "lstm_unbatched": ("lstm", 3, 4, 1, False, False, False, True, False, 4, 1),
    "gru": ("gru", 3, 4, 1, False, False, False, True, True, 4, 2),
    "gru_2layer_bidir_batch_first": ("gru", 3, 4, 2, True, True, False, True, True, 4, 2),
}


def rnn_param_shapes(kind, n_in, hidden, layers, bidir, bias):
    gates = {"rnn": 1, "lstm": 4, "gru": 3}[kind] * hidden
    dirs = 2 if bidir else 1
    shapes = []
    for layer in range(layers):
        width = n_in if layer == 0 else dirs * hidden
        for _ in range(dirs):
            shapes += [(gates, width), (gates, hidden)]
            if bias:
                shapes += [(gates,), (gates,)]
    return shapes


def np_recurrent(kind, layers, bidir, batch_first, relu, bias, batched):
    lstm = kind == "lstm"

    def fn(x, *rest):
        nstates = 2 if lstm else 1
        states, params = list(rest[:nstates]), iter(rest[nstates:])
        if not batched:
            x, states = x[:, None], [s[:, None] for s in states]
        elif batch_first:
            x = x.swapaxes(0, 1)
        dirs = 2 if bidir else 1
        inp, hs, cs = x, [], []
        for layer in range(layers):
            outs = []
            for d in range(dirs):
                w_ih, w_hh = next(params), next(params)
                b_ih, b_hh = (next(params), next(params)) if bias else (0.0, 0.0)
                at = layer * dirs + d
                h = states[0][at]
                c = states[1][at] if lstm else None
                ys = [None] * inp.shape[0]
                for t in (reversed(range(inp.shape[0])) if d else range(inp.shape[0])):
                    gi, gh = inp[t] @ w_ih.T + b_ih, h @ w_hh.T + b_hh
                    if kind == "rnn":
                        h = np.maximum(gi + gh, 0) if relu else np.tanh(gi + gh)
                    elif lstm:
                        i, f, g, o = np.split(gi + gh, 4, axis=-1)
                        c = np_sigmoid(f) * c + np_sigmoid(i) * np.tanh(g)
                        h = np_sigmoid(o) * np.tanh(c)
                    else:
                        ir, iz, inn = np.split(gi, 3, axis=-1)
                        hr, hz, hn = np.split(gh, 3, axis=-1)
                        r, z = np_sigmoid(ir + hr), np_sigmoid(iz + hz)
                        n = np.tanh(inn + r * hn)
                        h = (1 - z) * n + z * h
                    ys[t] = h
                outs.append(np.stack(ys))
                hs.append(h)
                cs.append(c)
            inp = np.concatenate(outs, axis=-1)
        out, extra = inp, {"h_n": np.stack(hs)}
        if lstm:
            extra["c_n"] = np.stack(cs)
        if not batched:
            out, extra = out[:, 0], {k: v[:, 0] for k, v in extra.items()}
        elif batch_first:
            out = out.swapaxes(0, 1)
        extra["output"] = out
        joined = np.concatenate([out.ravel()] + [extra[k].ravel() for k in ("h_n", "c_n") if k in extra])
        return joined, extra

    return fn


def torch_recurrent(kind, n_in, hidden, layers, bidir, batch_first, relu, bias):
    lstm = kind == "lstm"

    def fn(T, x, *rest):
        nstates = 2 if lstm else 1
        states, params = rest[:nstates], rest[nstates:]
        kw = dict(num_layers=layers, bidirectional=bidir, batch_first=batch_first, bias=bias, dtype=x.dtype)
        if kind == "rnn":
            kw["nonlinearity"] = "relu" if relu else "tanh"
        m = {"rnn": T.nn.RNN, "lstm": T.nn.LSTM, "gru": T.nn.GRU}[kind](n_in, hidden, **kw)
        names = [name for name, _ in m.named_parameters()]
        hx = tuple(states) if lstm else states[0]
        out, hn = T.func.functional_call(m, dict(zip(names, params)), (x, hx))
        extra = {"output": out, "h_n": hn[0] if lstm else hn}
        if lstm:
            extra["c_n"] = hn[1]
        joined = T.cat([out.reshape(-1)] + [extra[k].reshape(-1) for k in ("h_n", "c_n") if k in extra])
        return joined, extra

    return fn


def rnn_cases(rng):
    c = {}
    for name, (kind, n_in, hidden, layers, bidir, batch_first, relu, bias, batched, steps, batch) in RNN_CASES.items():
        dirs = 2 if bidir else 1
        if not batched:
            x_shape, state_shape = (steps, n_in), (layers * dirs, hidden)
        elif batch_first:
            x_shape, state_shape = (batch, steps, n_in), (layers * dirs, batch, hidden)
        else:
            x_shape, state_shape = (steps, batch, n_in), (layers * dirs, batch, hidden)
        inputs = [uniform(rng, x_shape, -1, 1)]
        inputs += [uniform(rng, state_shape, -0.5, 0.5) for _ in range(2 if kind == "lstm" else 1)]
        inputs += [uniform(rng, s, -0.5, 0.5) for s in rnn_param_shapes(kind, n_in, hidden, layers, bidir, bias)]
        c[name] = (inputs, list(range(len(inputs))),
                   np_recurrent(kind, layers, bidir, batch_first, relu, bias, batched),
                   torch_recurrent(kind, n_in, hidden, layers, bidir, batch_first, relu, bias))
    return c


# ---------------------------------------------------------------------------
# Block-quantized weights, as ggml lays out and dequantizes them. The
# dequantization computes in float32 and rounds after every operation, as
# ggml does, so mimir must match it bit for bit.

QUANT_BLOCKS = {"q8_0": (32, 34), "q4_K": (256, 144), "q6_K": (256, 210)}


def f16_scales(rng, n, signed=True):
    """n float16 scales near 1/64, as little-endian bytes [n, 2]."""
    s = rng.choice([1 / 16, 1 / 32, 1 / 64, 1 / 128, 3 / 256], size=n)
    if signed:
        s = s * rng.choice([-1, 1], size=n)
    return s.astype("<f2").view(np.uint8).reshape(n, 2)


def quant_weights(rng, kind, rows, k):
    """Random blocks of rows of k values, as uint8 [rows, bytes]: random
    quants and finite scales."""
    bs, bb = QUANT_BLOCKS[kind]
    b = rng.integers(0, 256, size=(rows * k // bs, bb), dtype=np.uint8)
    if kind == "q8_0":
        b[:, 0:2] = f16_scales(rng, len(b))
    elif kind == "q4_K":
        b[:, 0:2] = f16_scales(rng, len(b))
        b[:, 2:4] = f16_scales(rng, len(b), signed=False)
    else:
        b[:, 208:210] = f16_scales(rng, len(b))
    return b.reshape(rows, -1)


def np_dequantize(kind, raw):
    """The float32 values of the uint8 blocks raw [rows, bytes]."""
    bs, bb = QUANT_BLOCKS[kind]
    rows = raw.shape[0]
    b = raw.reshape(-1, bb)
    half = lambda lo: b[:, lo:lo + 2].copy().view("<f2").astype(F32)  # noqa: E731
    if kind == "q8_0":
        y = b[:, 2:].view(np.int8).astype(F32) * half(0)
    elif kind == "q4_K":
        d, dmin, q = half(0), half(2), b[:, 4:16].astype(np.int32)
        sc = np.concatenate([q[:, 0:4] & 63, (q[:, 8:12] & 15) | (q[:, 0:4] >> 6) << 4], axis=1)
        m = np.concatenate([q[:, 4:8] & 63, (q[:, 8:12] >> 4) | (q[:, 4:8] >> 6) << 4], axis=1)
        qs = b[:, 16:].reshape(-1, 4, 32).astype(np.int32)
        nib = np.stack([qs & 15, qs >> 4], axis=2).reshape(-1, 8, 32)  # sub-block 2j, 2j+1
        d1 = (d * sc.astype(F32))[:, :, None]
        m1 = (dmin * m.astype(F32))[:, :, None]
        y = d1 * nib.astype(F32) - m1
    else:
        ql, qh = b[:, 0:128].astype(np.int32), b[:, 128:192].astype(np.int32)
        sc, d = b[:, 192:208].view(np.int8).astype(F32), half(208)
        y = np.empty((len(b), 2, 4, 32), dtype=F32)
        for n in range(2):
            for k in range(4):
                lo = ql[:, 64 * n + 32 * (k % 2):64 * n + 32 * (k % 2) + 32] >> (4 * (k // 2)) & 15
                hi = qh[:, 32 * n:32 * n + 32] >> (2 * k) & 3
                q = ((lo | hi << 4) - 32).astype(F32)
                s = np.repeat(sc[:, 8 * n + 2 * k:8 * n + 2 * k + 2], 16, axis=1)
                y[:, n, k] = (d * s) * q
    return y.reshape(rows, -1).astype(F32)


def quantizable(rng, shape, block=32):
    """Values that int8 rounding in blocks of block keeps exactly: integers
    over 128, with ±127/128 in every block."""
    x = rng.integers(-127, 128, size=shape).astype(F32) / 128
    flat = x.reshape(-1, block)
    flat[np.arange(len(flat)), rng.integers(0, block, size=len(flat))] = rng.choice([-1, 1], size=len(flat)) * 127 / 128
    return flat.reshape(shape)


def quant_cases(rng):
    """Dequantizations, with diff None, and products x @ W^T by quantized
    weights, as gradient cases for x."""
    c = {}
    for kind, rows, k in [("q8_0", 4, 64), ("q4_K", 3, 512), ("q6_K", 3, 512)]:
        raw = quant_weights(rng, kind, rows, k)
        c[f"{kind}_dequantize"] = ([raw], None, lambda w, kind=kind: np_dequantize(kind, w), None)
    for kind in QUANT_BLOCKS:
        x, raw = quantizable(rng, (2, 3, 512)), quant_weights(rng, kind, 5, 512)
        c[f"{kind}_matmul"] = (
            [x, raw], [0],
            lambda x, w, kind=kind: x @ np_dequantize(kind, w).astype(np.float64).T,
            lambda T, x, w, kind=kind: x @ T.from_numpy(np_dequantize(kind, w.numpy())).to(x.dtype).T)
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
        dtype = {np.dtype(F32): "F32", np.dtype(I32): "I32", np.dtype(np.uint8): "U8"}[a.dtype]
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


def widen(inputs):
    return [a.astype(np.float64) if a.dtype == F32 else a for a in inputs]


def split_extra(result):
    """Oracles of stateful layers return (out, {name: extra result})."""
    if isinstance(result, tuple):
        return result
    return result, {}


def numeric_grads(np_fn, inputs, gout, diff):
    """Central differences of sum(np_fn(inputs) * gout) in float64."""
    wide = widen(inputs)
    loss = lambda xs: float((split_extra(np_fn(*xs))[0] * gout).sum())  # noqa: E731
    grads = []
    for i in diff:
        g = np.zeros_like(wide[i])
        for idx in np.ndindex(*wide[i].shape):
            h = 1e-6 * max(1.0, abs(wide[i][idx]))
            up = [a.copy() for a in wide]
            down = [a.copy() for a in wide]
            up[i][idx] += h
            down[i][idx] -= h
            g[idx] = (loss(up) - loss(down)) / (2 * h)
        grads.append(g)
    return grads


def evaluate_grad(oracle, inputs, diff, np_fn, torch_fn, gout_rng):
    out, extra = split_extra(np_fn(*widen(inputs)))
    gout = gout_rng.uniform(-1, 1, size=np.shape(out)).astype(F32)
    if oracle == "torch":
        import torch

        xs = [torch.tensor(a, dtype=torch.float64, requires_grad=i in diff) if a.dtype == F32 else torch.from_numpy(a)
              for i, a in enumerate(inputs)]
        out, extra = split_extra(torch_fn(torch, *xs))
        out.backward(torch.from_numpy(gout.astype(np.float64)))
        grads = [xs[i].grad.numpy() for i in diff]
        out = out.detach().numpy()
        extra = {k: v.detach().numpy() for k, v in extra.items()}
    else:
        grads = numeric_grads(np_fn, inputs, gout.astype(np.float64), diff)
    result = {"out": np.asarray(out), "gout": gout}
    result.update({f"grad{i}": g for i, g in zip(diff, grads)})
    result.update(extra)
    return {k: v.astype(F32) if v.dtype == np.float64 else v for k, v in result.items()}


# ---------------------------------------------------------------------------
# Attention and transformer layers (attention.safetensors, package nn).

def np_sdpa(q, k, v, mask=None, causal=False, offset=0, scale=None):
    """Attention over [..., H, Q, D] queries and [..., HK, K, D] keys, with
    each key head repeated for H/HK query heads; key j is hidden from query
    i when causal and j > i + offset."""
    if q.shape[-3] != k.shape[-3]:
        g = q.shape[-3] // k.shape[-3]
        k, v = np.repeat(k, g, axis=-3), np.repeat(v, g, axis=-3)
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])
    s = q @ np.swapaxes(k, -1, -2) * scale
    if mask is not None:
        s = s + mask
    if causal:
        i, j = np.arange(q.shape[-2])[:, None], np.arange(k.shape[-2])[None, :]
        s = np.where(j > i + offset, -np.inf, s)
    return np_softmax(s, -1) @ v


def bool_mask(m):
    """The additive form of a mask that is non-zero where a query attends."""
    return np.where(m != 0, 0.0, -np.inf)


def torch_causal(T, q, k, offset=0):
    """The boolean attn_mask of F.scaled_dot_product_attention that lets query
    i attend to keys j <= i + offset."""
    return T.ones(q.shape[-2], k.shape[-2], dtype=T.bool).tril(diagonal=offset)


def np_rope(x, dim, base, offset, interleaved=False, scaling=None):
    half = dim // 2
    inv = base ** (-np.arange(half) * 2.0 / dim)
    if scaling is not None:
        inv = scaling(inv)
    ang = (offset + np.arange(x.shape[-2]))[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    rot, rest = x[..., :dim], x[..., dim:]
    if interleaved:
        a, b = rot[..., 0::2], rot[..., 1::2]
        out = np.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(rot.shape)
    else:
        a, b = rot[..., :half], rot[..., half:]
        out = np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)
    return np.concatenate([out, rest], axis=-1)


def torch_rope(T, x, dim, base, offset, interleaved=False, scaling=None):
    half = dim // 2
    inv = base ** (-np.arange(half) * 2.0 / dim)
    if scaling is not None:
        inv = scaling(inv)
    ang = T.from_numpy((offset + np.arange(x.shape[-2]))[:, None] * inv[None, :]).to(x.dtype)
    cos, sin = ang.cos(), ang.sin()
    rot, rest = x[..., :dim], x[..., dim:]
    if interleaved:
        a, b = rot[..., 0::2], rot[..., 1::2]
        out = T.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).reshape(rot.shape)
    else:
        a, b = rot[..., :half], rot[..., half:]
        out = T.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
    return T.cat([out, rest], dim=-1)


def llama3_scaling(factor, low, high, original):
    def scale(inv):
        wave = 2 * math.pi / inv
        smooth = (original / wave - low) / (high - low)
        return np.where(wave < original / high, inv,
                        np.where(wave > original / low, inv / factor, (1 - smooth) * inv / factor + smooth * inv))
    return scale


def np_rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def heads(x, n):
    """[B, S, n*d] as [B, n, S, d]."""
    b, s, e = x.shape
    return np.swapaxes(x.reshape(b, s, n, e // n), 1, 2)


def merge(x):
    b, n, s, d = x.shape
    return np.swapaxes(x, 1, 2).reshape(b, s, n * d)


def np_mha(q, k, v, w, b, ow, ob, n, causal=False):
    e = w.shape[1]
    proj = [heads(x @ w[i * e:(i + 1) * e].T + b[i * e:(i + 1) * e], n) for i, x in enumerate((q, k, v))]
    return merge(np_sdpa(*proj, causal=causal)) @ ow.T + ob


def torch_mha(T, q, k, v, w, b, ow, ob, n, causal=False):
    m = T.nn.MultiheadAttention(w.shape[1], n, batch_first=True, dtype=q.dtype)
    params = {"in_proj_weight": w, "in_proj_bias": b, "out_proj.weight": ow, "out_proj.bias": ob}
    # The boolean attn_mask of nn.MultiheadAttention is True where a query
    # may not attend.
    mask = ~torch_causal(T, q, k) if causal else None
    return T.func.functional_call(m, params, (q, k, v), {"attn_mask": mask, "need_weights": False})[0]


def np_gpt2_block(x, ln1w, ln1b, wqkv, bqkv, wo, bo, ln2w, ln2b, w1, b1, w2, b2, n=2):
    e = x.shape[-1]
    qkv = np_layer_norm(x, ln1w, ln1b) @ wqkv.T + bqkv
    q, k, v = (heads(qkv[..., i * e:(i + 1) * e], n) for i in range(3))
    x = x + merge(np_sdpa(q, k, v, causal=True)) @ wo.T + bo
    h = np_layer_norm(x, ln2w, ln2b)
    return x + np_gelu(h @ w1.T + b1) @ w2.T + b2


def torch_gpt2_block(T, x, ln1w, ln1b, wqkv, bqkv, wo, bo, ln2w, ln2b, w1, b1, w2, b2, n=2):
    F = T.nn.functional
    e = x.shape[-1]
    split = lambda y: y.unflatten(-1, (n, e // n)).transpose(1, 2)  # noqa: E731
    qkv = F.linear(F.layer_norm(x, (e,), ln1w, ln1b, eps=1e-5), wqkv, bqkv)
    q, k, v = (split(qkv[..., i * e:(i + 1) * e]) for i in range(3))
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).flatten(-2)
    x = x + F.linear(a, wo, bo)
    h = F.layer_norm(x, (e,), ln2w, ln2b, eps=1e-5)
    return x + F.linear(F.gelu(F.linear(h, w1, b1), approximate="tanh"), w2, b2)


def np_llama_block(x, n1, wqkv, wo, n2, wg, wu, wd, n=4, nkv=2, eps=1e-5, base=10000.0):
    d = x.shape[-1] // n
    qkv = np_rms_norm(x, n1, eps) @ wqkv.T
    q = heads(qkv[..., :n * d], n)
    k = heads(qkv[..., n * d:(n + nkv) * d], nkv)
    v = heads(qkv[..., (n + nkv) * d:], nkv)
    q, k = np_rope(q, d, base, 0), np_rope(k, d, base, 0)
    x = x + merge(np_sdpa(q, k, v, causal=True)) @ wo.T
    h = np_rms_norm(x, n2, eps)
    g = h @ wg.T
    return x + (g * np_sigmoid(g) * (h @ wu.T)) @ wd.T


def torch_llama_block(T, x, n1, wqkv, wo, n2, wg, wu, wd, n=4, nkv=2, eps=1e-5, base=10000.0):
    F = T.nn.functional
    e = x.shape[-1]
    d = e // n
    split = lambda y, m: y.unflatten(-1, (m, d)).transpose(1, 2)  # noqa: E731
    qkv = F.linear(F.rms_norm(x, (e,), n1, eps=eps), wqkv)
    q = split(qkv[..., :n * d], n)
    k = split(qkv[..., n * d:(n + nkv) * d], nkv)
    v = split(qkv[..., (n + nkv) * d:], nkv)
    q, k = torch_rope(T, q, d, base, 0), torch_rope(T, k, d, base, 0)
    k, v = k.repeat_interleave(n // nkv, dim=1), v.repeat_interleave(n // nkv, dim=1)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).flatten(-2)
    x = x + F.linear(a, wo)
    h = F.rms_norm(x, (e,), n2, eps=eps)
    return x + F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wu), wd)


def np_encoder_layer(x, w, b, ow, ob, w1, b1, w2, b2, n1w, n1b, n2w, n2b, n=2):
    x = np_layer_norm(x + np_mha(x, x, x, w, b, ow, ob, n), n1w, n1b)
    return np_layer_norm(x + np.maximum(x @ w1.T + b1, 0) @ w2.T + b2, n2w, n2b)


def torch_encoder_layer(T, x, w, b, ow, ob, w1, b1, w2, b2, n1w, n1b, n2w, n2b, n=2):
    layer = T.nn.TransformerEncoderLayer(x.shape[-1], n, w1.shape[0], dropout=0.0, batch_first=True, dtype=x.dtype)
    layer.eval()
    params = {"self_attn.in_proj_weight": w, "self_attn.in_proj_bias": b, "self_attn.out_proj.weight": ow,
              "self_attn.out_proj.bias": ob, "linear1.weight": w1, "linear1.bias": b1, "linear2.weight": w2,
              "linear2.bias": b2, "norm1.weight": n1w, "norm1.bias": n1b, "norm2.weight": n2w, "norm2.bias": n2b}
    return T.func.functional_call(layer, params, (x,))


def attention_cases(rng):
    """Attention, its layers and transformer blocks, as gradient cases."""
    c = {}

    def case(name, inputs, diff, np_fn, torch_fn):
        c[name] = (inputs, diff, np_fn, torch_fn)

    def u(*shape, lo=-1.0, hi=1.0):
        return uniform(rng, shape, lo, hi)

    F = lambda T: T.nn.functional  # noqa: E731
    case("sdpa", [u(2, 3, 4, 8), u(2, 3, 6, 8), u(2, 3, 6, 5)], [0, 1, 2],
         lambda q, k, v: np_sdpa(q, k, v), lambda T, q, k, v: F(T).scaled_dot_product_attention(q, k, v))
    case("sdpa_causal", [u(2, 2, 5, 4), u(2, 2, 5, 4), u(2, 2, 5, 4)], [0, 1, 2],
         lambda q, k, v: np_sdpa(q, k, v, causal=True),
         lambda T, q, k, v: F(T).scaled_dot_product_attention(q, k, v, is_causal=True))
    mask = u(1, 1, 4, 6, lo=-2, hi=2)
    mask[..., 2] = -np.inf  # every row keeps other keys
    case("sdpa_mask", [u(2, 2, 4, 8), u(2, 2, 6, 8), u(2, 2, 6, 8), mask], [0, 1, 2],
         lambda q, k, v, m: np_sdpa(q, k, v, mask=m),
         lambda T, q, k, v, m: F(T).scaled_dot_product_attention(q, k, v, attn_mask=m))
    keep = (rng.uniform(size=(2, 1, 4, 6)) < 0.6).astype(F32)
    keep[..., 0] = 1
    case("sdpa_bool_mask", [u(2, 2, 4, 8), u(2, 2, 6, 8), u(2, 2, 6, 8), keep], [0, 1, 2],
         lambda q, k, v, m: np_sdpa(q, k, v, mask=bool_mask(m)),
         lambda T, q, k, v, m: F(T).scaled_dot_product_attention(q, k, v, attn_mask=m != 0))
    case("sdpa_gqa", [u(2, 4, 3, 8), u(2, 2, 5, 8), u(2, 2, 5, 8)], [0, 1, 2],
         lambda q, k, v: np_sdpa(q, k, v, causal=True, offset=2),
         lambda T, q, k, v: F(T).scaled_dot_product_attention(
             q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), attn_mask=torch_causal(T, q, k, 2)))
    case("sdpa_scale", [u(1, 2, 4, 8), u(1, 2, 6, 8), u(1, 2, 6, 3)], [0, 1, 2],
         lambda q, k, v: np_sdpa(q, k, v, scale=0.3),
         lambda T, q, k, v: F(T).scaled_dot_product_attention(q, k, v, scale=0.3))
    case("sdpa_causal_offset", [u(1, 2, 3, 4), u(1, 2, 7, 4), u(1, 2, 7, 4)], [0, 1, 2],
         lambda q, k, v: np_sdpa(q, k, v, causal=True, offset=4),
         lambda T, q, k, v: F(T).scaled_dot_product_attention(q, k, v, attn_mask=torch_causal(T, q, k, 4)))

    case("multihead_attention", [u(2, 5, 8), u(24, 8, lo=-0.5, hi=0.5), u(24), u(8, 8, lo=-0.5, hi=0.5), u(8)],
         [0, 1, 2, 3, 4],
         lambda x, w, b, ow, ob: np_mha(x, x, x, w, b, ow, ob, 2),
         lambda T, x, w, b, ow, ob: torch_mha(T, x, x, x, w, b, ow, ob, 2))
    case("multihead_attention_causal",
         [u(2, 4, 8), u(2, 6, 8), u(2, 6, 8), u(24, 8, lo=-0.5, hi=0.5), u(24), u(8, 8, lo=-0.5, hi=0.5), u(8)],
         [0, 1, 2, 3, 4, 5, 6],
         lambda q, k, v, w, b, ow, ob: np_mha(q, k, v, w, b, ow, ob, 4, causal=True),
         lambda T, q, k, v, w, b, ow, ob: torch_mha(T, q, k, v, w, b, ow, ob, 4, causal=True))

    case("rms_norm", [u(3, 4, 8, lo=-2, hi=2), u(8, lo=0.5, hi=1.5)], [0, 1],
         lambda x, w: np_rms_norm(x, w, 1e-6), lambda T, x, w: F(T).rms_norm(x, (8,), w, eps=1e-6))
    case("rms_norm_default_eps", [u(2, 6, lo=-0.01, hi=0.01), u(6, lo=0.5, hi=1.5)], [0, 1],
         lambda x, w: np_rms_norm(x, w, 2.0**-23), lambda T, x, w: F(T).rms_norm(x, (6,), w, eps=2.0**-23))
    case("rope_neox", [u(2, 3, 5, 8, lo=-2, hi=2)], [0],
         lambda x: np_rope(x, 8, 10000.0, 3), lambda T, x: torch_rope(T, x, 8, 10000.0, 3))
    case("rope_interleaved", [u(2, 3, 5, 8, lo=-2, hi=2)], [0],
         lambda x: np_rope(x, 8, 10000.0, 3, interleaved=True),
         lambda T, x: torch_rope(T, x, 8, 10000.0, 3, interleaved=True))
    scaling = llama3_scaling(8.0, 1.0, 4.0, 64)
    case("rope_partial_scaled", [u(1, 2, 4, 12, lo=-2, hi=2)], [0],
         lambda x: np_rope(x, 8, 10000.0, 2, scaling=scaling),
         lambda T, x: torch_rope(T, x, 8, 10000.0, 2, scaling=scaling))

    def ln():
        return [u(8, lo=0.5, hi=1.5), u(8, lo=-0.5, hi=0.5)]

    gpt2 = [u(2, 5, 8), *ln(), u(24, 8, lo=-0.5, hi=0.5), u(24, lo=-0.2, hi=0.2), u(8, 8, lo=-0.5, hi=0.5),
            u(8, lo=-0.2, hi=0.2), *ln(), u(32, 8, lo=-0.5, hi=0.5), u(32, lo=-0.2, hi=0.2),
            u(8, 32, lo=-0.3, hi=0.3), u(8, lo=-0.2, hi=0.2)]
    case("decoder_block_gpt2", gpt2, list(range(len(gpt2))), np_gpt2_block, torch_gpt2_block)
    llama = [u(2, 5, 16), u(16, lo=0.5, hi=1.5), u(32, 16, lo=-0.4, hi=0.4), u(16, 16, lo=-0.4, hi=0.4),
             u(16, lo=0.5, hi=1.5), u(24, 16, lo=-0.4, hi=0.4), u(24, 16, lo=-0.4, hi=0.4), u(16, 24, lo=-0.3, hi=0.3)]
    case("decoder_block_llama", llama, list(range(len(llama))), np_llama_block, torch_llama_block)
    enc = [u(2, 5, 8), u(24, 8, lo=-0.5, hi=0.5), u(24, lo=-0.2, hi=0.2), u(8, 8, lo=-0.5, hi=0.5),
           u(8, lo=-0.2, hi=0.2), u(16, 8, lo=-0.5, hi=0.5), u(16, lo=-0.2, hi=0.2), u(8, 16, lo=-0.4, hi=0.4),
           u(8, lo=-0.2, hi=0.2), *ln(), *ln()]
    case("transformer_encoder_layer", enc, list(range(len(enc))), np_encoder_layer, torch_encoder_layer)
    return c


# ---------------------------------------------------------------------------
# Sorts, top-k, Fourier transforms and linear algebra (linalg.safetensors,
# package tensor). Complex tensors are real ones with a last dimension of 2,
# as torch.view_as_real shows them. Functions of eigenvectors and singular
# vectors use their squares, which do not depend on their signs, and inputs
# of Cholesky and eigh are made symmetric inside the case.


def np_complex(x):
    return x[..., 0] + 1j * x[..., 1]


def np_real_view(c):
    return np.stack([c.real, c.imag], axis=-1)


def np_spd(x, shift):
    return x @ np.swapaxes(x, -1, -2) + shift * np.eye(x.shape[-1])


def torch_spd(T, x, shift):
    return x @ x.mT + shift * T.eye(x.shape[-1], dtype=x.dtype)


def np_shift(x, by):
    return x + by * np.eye(x.shape[-1])


def torch_shift(T, x, by):
    return x + by * T.eye(x.shape[-1], dtype=x.dtype)


def np_flat(*arrays):
    return np.concatenate([np.ravel(a) for a in arrays])


def torch_flat(T, *ts):
    return T.cat([t.reshape(-1) for t in ts])


def np_svd_parts(x):
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    return np_flat(s, u * u, vh * vh)


def torch_svd_parts(T, x):
    u, s, vh = T.linalg.svd(x, full_matrices=False)
    return torch_flat(T, s, u * u, vh * vh)


def linalg_cases(rng):
    """Cases as gradient cases. A case without differentiable inputs checks
    its output alone."""
    c = {}

    def case(name, inputs, diff, np_fn, torch_fn):
        c[name] = (inputs, diff, np_fn, torch_fn)

    def distinct(*shape):
        return (rng.permutation(int(np.prod(shape))).reshape(shape) * 0.25 - 2).astype(F32)

    def ties(*shape):
        return np.round(uniform(rng, shape, -2, 2)).astype(F32)

    case("sort_values", [distinct(4, 7)], [0],
         lambda x: np.sort(x, axis=1), lambda T, x: T.sort(x, dim=1, stable=True).values)
    case("sort_descending_indices", [ties(4, 9)], [],
         lambda x: np.argsort(-x, axis=1, kind="stable").astype(I32),
         lambda T, x: T.sort(x, dim=1, descending=True, stable=True).indices.to(T.int32))
    case("argsort_axis0", [ties(6, 3)], [],
         lambda x: np.argsort(x, axis=0, kind="stable").astype(I32),
         lambda T, x: T.argsort(x, dim=0, stable=True).to(T.int32))
    case("topk_values", [distinct(5, 6)], [0],
         lambda x: -np.sort(-x, axis=1)[:, :3], lambda T, x: T.topk(x, 3, dim=1).values)
    case("topk_smallest_indices", [distinct(7, 4)], [],
         lambda x: np.argsort(x, axis=0)[:2].astype(I32),
         lambda T, x: T.topk(x, 2, dim=0, largest=False).indices.to(T.int32))

    case("fft", [uniform(rng, (3, 8, 2), -1, 1)], [0],
         lambda x: np_real_view(np.fft.fft(np_complex(x), axis=1)),
         lambda T, x: T.view_as_real(T.fft.fft(T.view_as_complex(x), dim=1)))
    case("fft_bluestein", [uniform(rng, (20, 3, 2), -1, 1)], [0],
         lambda x: np_real_view(np.fft.fft(np_complex(x), axis=0)),
         lambda T, x: T.view_as_real(T.fft.fft(T.view_as_complex(x), dim=0)))
    case("ifft", [uniform(rng, (2, 6, 2), -1, 1)], [0],
         lambda x: np_real_view(np.fft.ifft(np_complex(x), axis=-1)),
         lambda T, x: T.view_as_real(T.fft.ifft(T.view_as_complex(x), dim=-1)))
    case("rfft", [uniform(rng, (3, 10), -1, 1)], [0],
         lambda x: np_real_view(np.fft.rfft(x, axis=-1)),
         lambda T, x: T.view_as_real(T.fft.rfft(x, dim=-1)))
    case("irfft", [uniform(rng, (2, 5, 2), -1, 1)], [0],
         lambda x: np.fft.irfft(np_complex(x), n=8, axis=-1),
         lambda T, x: T.fft.irfft(T.view_as_complex(x), n=8, dim=-1))
    case("irfft_odd_padded", [uniform(rng, (4, 2, 2), -1, 1)], [0],
         lambda x: np.fft.irfft(np_complex(x), n=9, axis=0),
         lambda T, x: T.fft.irfft(T.view_as_complex(x), n=9, dim=0))

    case("cholesky", [uniform(rng, (2, 4, 4), -1, 1)], [0],
         lambda x: np.linalg.cholesky(np_spd(x, 2)), lambda T, x: T.linalg.cholesky(torch_spd(T, x, 2)))
    case("solve", [uniform(rng, (2, 4, 4), -1, 1), uniform(rng, (2, 4, 3), -1, 1)], [0, 1],
         lambda a, b: np.linalg.solve(np_shift(a, 3), b), lambda T, a, b: T.linalg.solve(torch_shift(T, a, 3), b))
    case("solve_vector", [uniform(rng, (4, 4), -1, 1), uniform(rng, (4,), -1, 1)], [0, 1],
         lambda a, b: np.linalg.solve(np_shift(a, 3), b), lambda T, a, b: T.linalg.solve(torch_shift(T, a, 3), b))
    case("triangular_solve_lower", [uniform(rng, (3, 3), -1, 1), uniform(rng, (3, 2), -1, 1)], [0, 1],
         lambda a, b: np.linalg.solve(np.tril(np_shift(a, 3)), b),
         lambda T, a, b: T.linalg.solve_triangular(torch_shift(T, a, 3), b, upper=False))
    case("triangular_solve_upper", [uniform(rng, (2, 3, 3), -1, 1), uniform(rng, (2, 3, 4), -1, 1)], [0, 1],
         lambda a, b: np.linalg.solve(np.triu(np_shift(a, 3)), b),
         lambda T, a, b: T.linalg.solve_triangular(torch_shift(T, a, 3), b, upper=True))
    case("inv", [uniform(rng, (3, 3), -1, 1)], [0],
         lambda x: np.linalg.inv(np_shift(x, 3)), lambda T, x: T.linalg.inv(torch_shift(T, x, 3)))
    case("det", [uniform(rng, (2, 3, 3), -1, 1)], [0],
         lambda x: np.linalg.det(np_shift(x, 1)), lambda T, x: T.linalg.det(torch_shift(T, x, 1)))
    slog = uniform(rng, (2, 4, 4), -2, 2)
    case("slogdet_sign", [slog], [],
         lambda x: np.linalg.slogdet(x)[0].astype(F32), lambda T, x: T.linalg.slogdet(x)[0])
    case("slogdet_logabsdet", [slog], [0],
         lambda x: np.linalg.slogdet(x)[1], lambda T, x: T.linalg.slogdet(x)[1])
    case("qr_tall", [uniform(rng, (5, 3), -1, 1)], [0],
         lambda x: np_flat(*np.linalg.qr(x)), lambda T, x: torch_flat(T, *T.linalg.qr(x)))
    case("qr_wide", [uniform(rng, (2, 3, 5), -1, 1)], [0],
         lambda x: np_flat(*np.linalg.qr(x)), lambda T, x: torch_flat(T, *T.linalg.qr(x)))
    case("svd_tall", [uniform(rng, (5, 3), -1, 1)], [0], np_svd_parts, torch_svd_parts)
    case("svd_wide", [uniform(rng, (2, 3, 4), -1, 1)], [0], np_svd_parts, torch_svd_parts)
    case("eigh", [uniform(rng, (4, 4), -2, 2)], [0],
         lambda x: np_flat(*(lambda w, v: (w, v * v))(*np.linalg.eigh(x + x.T))),
         lambda T, x: (lambda r: torch_flat(T, r.eigenvalues, r.eigenvectors ** 2))(T.linalg.eigh(x + x.mT)))
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out", default=os.path.join("testdata", "golden"))
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()

    metadata = {"oracle": args.oracle, "numpy": np.__version__, "seed": str(args.seed)}
    if args.oracle == "torch":
        import torch

        torch.set_num_threads(1)
        metadata["torch"] = torch.__version__

    files = {}
    tensors = {}
    for name, (inputs, np_fn, torch_fn) in cases(np.random.default_rng(args.seed)).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        out = evaluate(args.oracle, inputs, np_fn, torch_fn)
        if out.dtype not in (np.dtype(F32), np.dtype(I32)):
            raise SystemExit(f"{name}: oracle returned {out.dtype}")
        tensors[f"{name}/out"] = out
    files["ops"] = tensors

    # Later files draw from their own streams so adding cases to one file
    # never changes another.
    for offset, (file, make) in enumerate([("grads", grad_cases), ("nn", nn_cases)], start=1):
        rng = np.random.default_rng(args.seed + offset)
        gout_rng = np.random.default_rng(args.seed + 100 + offset)
        tensors = {}
        for name, (inputs, diff, np_fn, torch_fn) in make(rng).items():
            for i, a in enumerate(inputs):
                tensors[f"{name}/in{i}"] = a
            for part, a in evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng).items():
                tensors[f"{name}/{part}"] = a
        files[file] = tensors
    files["optim"] = optim_tensors(args.oracle, np.random.default_rng(args.seed + 3))

    rng = np.random.default_rng(args.seed + 6)
    gout_rng = np.random.default_rng(args.seed + 100 + 6)
    tensors = {}
    for name, (inputs, diff, np_fn, torch_fn) in scan_cases(rng).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        if diff:
            parts = evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng)
        else:
            parts = {"out": evaluate(args.oracle, inputs, np_fn, torch_fn)}
        for part, a in parts.items():
            tensors[f"{name}/{part}"] = a
    files["scan"] = tensors

    # Recurrent layers, from their own streams at seed offset 5.
    rng, gout_rng = np.random.default_rng(args.seed + 5), np.random.default_rng(args.seed + 105)
    tensors = {}
    for name, (inputs, diff, np_fn, torch_fn) in rnn_cases(rng).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        for part, a in evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng).items():
            tensors[f"{name}/{part}"] = a
    files["rnn"] = tensors

    # Block-quantized weights, from their own streams at seed offset 16.
    rng, gout_rng = np.random.default_rng(args.seed + 16), np.random.default_rng(args.seed + 116)
    tensors = {}
    for name, (inputs, diff, np_fn, torch_fn) in quant_cases(rng).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        if diff is None:
            tensors[f"{name}/out"] = evaluate(args.oracle, inputs, np_fn, torch_fn)
            continue
        for part, a in evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng).items():
            tensors[f"{name}/{part}"] = a
    files["quant"] = tensors

    # Attention and transformer layers, from their own streams at seed offset 4.
    rng, gout_rng = np.random.default_rng(args.seed + 4), np.random.default_rng(args.seed + 100 + 4)
    tensors = {}
    for name, (inputs, diff, np_fn, torch_fn) in attention_cases(rng).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        for part, a in evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng).items():
            tensors[f"{name}/{part}"] = a
    files["attention"] = tensors

    # Sorts, Fourier transforms and linear algebra, from their own streams at
    # seed offset 8.
    rng, gout_rng = np.random.default_rng(args.seed + 8), np.random.default_rng(args.seed + 108)
    tensors = {}
    for name, (inputs, diff, np_fn, torch_fn) in linalg_cases(rng).items():
        for i, a in enumerate(inputs):
            tensors[f"{name}/in{i}"] = a
        if diff:
            parts = evaluate_grad(args.oracle, inputs, diff, np_fn, torch_fn, gout_rng)
        else:
            parts = {"out": evaluate(args.oracle, inputs, np_fn, torch_fn)}
        for part, a in parts.items():
            tensors[f"{name}/{part}"] = a
    files["linalg"] = tensors

    os.makedirs(args.out, exist_ok=True)
    for file, tensors in files.items():
        path = os.path.join(args.out, file + ".safetensors")
        write_safetensors(path, tensors, metadata)
        print(f"wrote {len(tensors)} tensors to {path} (oracle {args.oracle})")


if __name__ == "__main__":
    main()
