#!/usr/bin/env python3
"""Generate the ONNX models that io/onnx's golden tests run.

    python3 tools/onnx/gen.py --out io/onnx/testdata
    pip install numpy torch onnx
    python3 tools/onnx/gen.py --oracle torch --out /tmp/onnx

Each model <name> is written as <name>.onnx, with <name>.safetensors
holding its inputs, in/<input>, and expected outputs, out/<output>:

    mlp          Gemm, Relu, Gemm and Softmax over a symbolic batch
    cnn          Conv, BatchNormalization, Relu, MaxPool, a grouped and
                 strided Conv, AveragePool, GlobalAveragePool, Flatten,
                 Gemm and Softmax
    decoder      ConvTranspose, Resize (nearest and linear), a reflecting
                 Pad, Conv and Sigmoid, with two outputs
    transformer  a two-layer causal transformer over int64 tokens:
                 embeddings by Gather, positions by Range, LayerNormalization,
                 attention from MatMul, Softmax and a Trilu mask, heads
                 split by Reshape and Transpose with targets computed by
                 Shape, Gather and Concat, and an Erf GELU

Oracles:
    numpy  writes each graph with the protocol buffer writer below, which
           needs only NumPy, and computes the outputs in float64 with
           NumPy, rounded to float32. These are the committed files.
    torch  builds the same networks with the same weights as PyTorch
           modules, exports them with torch.onnx.export at opset 17 with
           dynamic batch and sequence axes, and records PyTorch's outputs.
           The graphs differ from the hand-written ones, as an exporter's
           do, but compute the same functions, so the tests check mimir
           against the models PyTorch writes:

           MIMIR_ONNX_DIR=/tmp/onnx go test -run TestGolden ./io/onnx

Weights and inputs come from a NumPy generator seeded with --seed plus
12, the offset of these files among mimir's golden data.
"""

import argparse
import json
import math
import os
import struct

import numpy as np

F32, I32, I64 = np.float32, np.int32, np.int64

# TensorProto.DataType
FLOAT, INT32, INT64, BOOL = 1, 6, 7, 9
ONNX_TYPE = {np.dtype(F32): FLOAT, np.dtype(I32): INT32, np.dtype(I64): INT64, np.dtype(np.bool_): BOOL}


# Protocol buffer encoding.


def varint(v):
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field, wire):
    return varint(field << 3 | wire)


def f_varint(field, v):
    return key(field, 0) + varint(int(v))


def f_bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return key(field, 2) + varint(len(b)) + b


def f_float(field, f):
    return key(field, 5) + struct.pack("<f", f)


def f_packed(field, values):
    return f_bytes(field, b"".join(varint(int(v)) for v in values))


def tensor_proto(name, a, typed=False):
    """A TensorProto of array a: raw_data, or the typed fields if typed."""
    a = np.asarray(a)
    out = b"".join(f_varint(1, d) for d in a.shape)
    out += f_varint(2, ONNX_TYPE[a.dtype])
    if typed and a.dtype == F32:
        out += f_bytes(4, a.astype("<f4").tobytes())  # float_data, packed
    elif typed and a.dtype == I64:
        out += f_packed(7, a.ravel())  # int64_data
    else:
        out += f_bytes(9, a.astype(a.dtype.newbyteorder("<")).tobytes())
    return out + f_bytes(8, name)


def attribute(name, v):
    out = f_bytes(1, name)
    if isinstance(v, bool) or isinstance(v, (int, np.integer)):
        return out + f_varint(20, 2) + f_varint(3, int(v))
    if isinstance(v, float):
        return out + f_varint(20, 1) + f_float(2, v)
    if isinstance(v, str):
        return out + f_varint(20, 3) + f_bytes(4, v)
    if isinstance(v, np.ndarray):
        return out + f_varint(20, 4) + f_bytes(5, tensor_proto("", v))
    if all(isinstance(x, (int, np.integer)) for x in v):
        return out + f_varint(20, 7) + f_packed(8, v)
    return out + f_varint(20, 6) + b"".join(f_float(7, x) for x in v)


def value_info(name, elem, dims):
    shape = b""
    for d in dims:
        dim = f_bytes(2, d) if isinstance(d, str) else f_varint(1, d)
        shape += f_bytes(1, dim)
    tensor_type = f_varint(1, elem) + f_bytes(2, shape)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, tensor_type))


class Graph:
    """An ONNX graph, written as a ModelProto by model()."""

    def __init__(self, name):
        self.name = name
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self.count = 0

    def fresh(self, base):
        self.count += 1
        return f"{base}_{self.count}"

    def input(self, name, dtype, dims):
        self.inputs.append(value_info(name, ONNX_TYPE[np.dtype(dtype)], dims))
        return name

    def output(self, name, dtype, dims):
        self.outputs.append(value_info(name, ONNX_TYPE[np.dtype(dtype)], dims))

    def init(self, name, a, typed=False):
        self.inits.append(tensor_proto(name, a, typed))
        return name

    def ints(self, *values):
        return self.init(self.fresh("const"), np.array(values, I64))

    def scalar(self, v):
        return self.init(self.fresh("const"), np.array(v, F32))

    def op(self, op, inputs, outputs=1, name=None, **attrs):
        if isinstance(outputs, int):
            outputs = [self.fresh(op.lower()) for _ in range(outputs)]
        node = b"".join(f_bytes(1, i) for i in inputs) + b"".join(f_bytes(2, o) for o in outputs)
        node += f_bytes(3, name or self.fresh(op)) + f_bytes(4, op)
        node += b"".join(f_bytes(5, attribute(k, v)) for k, v in sorted(attrs.items()))
        self.nodes.append(node)
        return outputs[0] if len(outputs) == 1 else outputs

    def model(self, opset=17):
        g = b"".join(f_bytes(1, n) for n in self.nodes) + f_bytes(2, self.name)
        g += b"".join(f_bytes(5, t) for t in self.inits)
        g += b"".join(f_bytes(11, v) for v in self.inputs) + b"".join(f_bytes(12, v) for v in self.outputs)
        return f_varint(1, 8) + f_bytes(2, "tools/onnx/gen.py") + f_bytes(7, g) + f_bytes(8, f_bytes(1, "") + f_varint(2, opset))


# NumPy references, in float64.


def conv2d(x, w, b=None, stride=(1, 1), pads=(0, 0, 0, 0), dil=(1, 1), groups=1):
    n, c, h, wd = x.shape
    m, cg, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    oh = (h + pads[0] + pads[2] - (kh - 1) * dil[0] - 1) // stride[0] + 1
    ow = (wd + pads[1] + pads[3] - (kw - 1) * dil[1] - 1) // stride[1] + 1
    out = np.zeros((n, m, oh, ow))
    mg = m // groups
    for g in range(groups):
        xs, ws = xp[:, g * cg : (g + 1) * cg], w[g * mg : (g + 1) * mg]
        for p in range(kh):
            for q in range(kw):
                i0, j0 = p * dil[0], q * dil[1]
                patch = xs[:, :, i0 : i0 + stride[0] * (oh - 1) + 1 : stride[0], j0 : j0 + stride[1] * (ow - 1) + 1 : stride[1]]
                out[:, g * mg : (g + 1) * mg] += np.einsum("nchw,mc->nmhw", patch, ws[:, :, p, q])
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def conv_transpose2d(x, w, b, stride, pads, out_pad):
    n, c, h, wd = x.shape
    _, m, kh, kw = w.shape
    full = np.zeros((n, m, (h - 1) * stride[0] + kh + out_pad[0], (wd - 1) * stride[1] + kw + out_pad[1]))
    for i in range(h):
        for j in range(wd):
            full[:, :, i * stride[0] : i * stride[0] + kh, j * stride[1] : j * stride[1] + kw] += np.einsum("nc,cmpq->nmpq", x[:, :, i, j], w)
    out = full[:, :, pads[0] : full.shape[2] - pads[2], pads[1] : full.shape[3] - pads[3]]
    return out + b.reshape(1, -1, 1, 1)


def max_pool(x, k, s):
    n, c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    out = np.full((n, c, oh, ow), -np.inf)
    for p in range(k):
        for q in range(k):
            out = np.maximum(out, x[:, :, p : p + s * (oh - 1) + 1 : s, q : q + s * (ow - 1) + 1 : s])
    return out


def avg_pool(x, k, s):
    n, c, h, w = x.shape
    oh, ow = (h - k) // s + 1, (w - k) // s + 1
    out = np.zeros((n, c, oh, ow))
    for p in range(k):
        for q in range(k):
            out += x[:, :, p : p + s * (oh - 1) + 1 : s, q : q + s * (ow - 1) + 1 : s]
    return out / (k * k)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def resize_linear(x, size):
    """Bilinear resize with half-pixel coordinates, clamped at the edges."""

    def axis(a, n_in, n_out, ax):
        src = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, n_in - 1)
        w = src - lo
        shape = [1] * a.ndim
        shape[ax] = n_out
        w = w.reshape(shape)
        return np.take(a, lo, axis=ax) * (1 - w) + np.take(a, hi, axis=ax) * w

    y = axis(x, x.shape[2], size[0], 2)
    return axis(y, x.shape[3], size[1], 3)


# Models. Each returns (graph, weights, inputs, outputs from NumPy).


def normal(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(F32)


def mlp(rng):
    w1, b1 = normal(rng, 32, 16, scale=0.25), normal(rng, 32, scale=0.1)
    w2, b2 = normal(rng, 10, 32, scale=0.2), normal(rng, 10, scale=0.1)
    x = normal(rng, 4, 16)
    g = Graph("mlp")
    g.input("x", F32, ["batch", 16])
    # Typed fields rather than raw_data, as some writers store them.
    h = g.op("Gemm", ["x", g.init("w1", w1, typed=True), g.init("b1", b1, typed=True)], transB=1)
    h = g.op("Relu", [h])
    h = g.op("Gemm", [h, g.init("w2", w2), g.init("b2", b2)], transB=1)
    g.op("Softmax", [h], ["probs"], axis=-1)
    g.output("probs", F32, ["batch", 10])

    def ref(x):
        h = np.maximum(x @ w1.T + b1, 0)
        return {"probs": softmax(h @ w2.T + b2)}

    return g, {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, {"x": x}, ref


def cnn(rng):
    p = {
        "c1.w": normal(rng, 8, 3, 3, 3, scale=0.3), "c1.b": normal(rng, 8, scale=0.1),
        "bn.scale": (1 + 0.1 * rng.standard_normal(8)).astype(F32), "bn.bias": normal(rng, 8, scale=0.1),
        "bn.mean": normal(rng, 8, scale=0.1), "bn.var": (1 + 0.2 * rng.random(8)).astype(F32),
        "c2.w": normal(rng, 16, 4, 3, 3, scale=0.3), "c2.b": normal(rng, 16, scale=0.1),
        "fc.w": normal(rng, 10, 16, scale=0.3), "fc.b": normal(rng, 10, scale=0.1),
    }
    x = normal(rng, 2, 3, 16, 16)
    g = Graph("cnn")
    g.input("x", F32, ["batch", 3, 16, 16])
    for k, v in p.items():
        g.init(k, v)
    h = g.op("Conv", ["x", "c1.w", "c1.b"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    h = g.op("BatchNormalization", [h, "bn.scale", "bn.bias", "bn.mean", "bn.var"], epsilon=1e-5)
    h = g.op("Relu", [h])
    h = g.op("MaxPool", [h], kernel_shape=[2, 2], strides=[2, 2])
    h = g.op("Conv", [h, "c2.w", "c2.b"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], group=2)
    h = g.op("Relu", [h])
    h = g.op("AveragePool", [h], kernel_shape=[2, 2], strides=[2, 2])
    h = g.op("GlobalAveragePool", [h])
    h = g.op("Flatten", [h], axis=1)
    h = g.op("Gemm", [h, "fc.w", "fc.b"], transB=1)
    g.op("Softmax", [h], ["probs"], axis=1)
    g.output("probs", F32, ["batch", 10])

    def ref(x):
        h = conv2d(x, p["c1.w"], p["c1.b"], pads=(1, 1, 1, 1))
        s = p["bn.scale"] / np.sqrt(p["bn.var"].astype(np.float64) + 1e-5)
        h = (h - p["bn.mean"].reshape(1, -1, 1, 1)) * s.reshape(1, -1, 1, 1) + p["bn.bias"].reshape(1, -1, 1, 1)
        h = max_pool(np.maximum(h, 0), 2, 2)
        h = np.maximum(conv2d(h, p["c2.w"], p["c2.b"], stride=(2, 2), pads=(1, 1, 1, 1), groups=2), 0)
        h = avg_pool(h, 2, 2).mean(axis=(2, 3))
        return {"probs": softmax(h @ p["fc.w"].T + p["fc.b"], axis=1)}

    return g, p, {"x": x}, ref


def decoder(rng):
    p = {
        "up.w": normal(rng, 4, 3, 3, 3, scale=0.3), "up.b": normal(rng, 3, scale=0.1),
        "out.w": normal(rng, 2, 3, 3, 3, scale=0.3), "out.b": normal(rng, 2, scale=0.1),
    }
    x = normal(rng, 1, 4, 5, 5)
    g = Graph("decoder")
    g.input("x", F32, ["batch", 4, 5, 5])
    for k, v in p.items():
        g.init(k, v)
    h = g.op("ConvTranspose", ["x", "up.w", "up.b"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])
    h = g.op("Resize", [h, "", g.init("scales", np.array([1, 1, 2, 2], F32))], mode="nearest",
             coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    h = g.op("Pad", [h, g.ints(0, 0, 1, 1, 0, 0, 1, 1)], mode="reflect")
    h = g.op("Conv", [h, "out.w", "out.b"], kernel_shape=[3, 3])
    g.op("Sigmoid", [h], ["image"])
    g.op("Resize", ["x", "", "", g.ints(1, 4, 7, 7)], ["smooth"], mode="linear", coordinate_transformation_mode="half_pixel")
    g.output("image", F32, ["batch", 2, 20, 20])
    g.output("smooth", F32, [1, 4, 7, 7])

    def ref(x):
        h = conv_transpose2d(x, p["up.w"], p["up.b"], (2, 2), (1, 1, 1, 1), (1, 1))
        h = h.repeat(2, axis=2).repeat(2, axis=3)
        h = np.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        h = conv2d(h, p["out.w"], p["out.b"])
        return {"image": 1 / (1 + np.exp(-h)), "smooth": resize_linear(x, (7, 7))}

    return g, p, {"x": x}, ref


VOCAB, DIM, HEADS, LAYERS, MAXLEN = 50, 32, 4, 2, 16


def transformer_weights(rng):
    p = {"wte": normal(rng, VOCAB, DIM, scale=0.5), "wpe": normal(rng, MAXLEN, DIM, scale=0.5)}
    for i in range(LAYERS):
        p.update({
            f"h{i}.ln1.g": (1 + 0.1 * rng.standard_normal(DIM)).astype(F32), f"h{i}.ln1.b": normal(rng, DIM, scale=0.1),
            f"h{i}.qkv.w": normal(rng, DIM, 3 * DIM, scale=DIM ** -0.5), f"h{i}.qkv.b": normal(rng, 3 * DIM, scale=0.1),
            f"h{i}.proj.w": normal(rng, DIM, DIM, scale=DIM ** -0.5), f"h{i}.proj.b": normal(rng, DIM, scale=0.1),
            f"h{i}.ln2.g": (1 + 0.1 * rng.standard_normal(DIM)).astype(F32), f"h{i}.ln2.b": normal(rng, DIM, scale=0.1),
            f"h{i}.fc.w": normal(rng, DIM, 4 * DIM, scale=DIM ** -0.5), f"h{i}.fc.b": normal(rng, 4 * DIM, scale=0.1),
            f"h{i}.out.w": normal(rng, 4 * DIM, DIM, scale=(4 * DIM) ** -0.5), f"h{i}.out.b": normal(rng, DIM, scale=0.1),
        })
    p.update({"lnf.g": np.ones(DIM, F32), "lnf.b": np.zeros(DIM, F32), "head.w": normal(rng, DIM, VOCAB, scale=DIM ** -0.5)})
    return p


def transformer(rng):
    p = transformer_weights(rng)
    tokens = rng.integers(0, VOCAB, size=(2, 7)).astype(I64)
    dh = DIM // HEADS
    g = Graph("transformer")
    g.input("tokens", I64, ["batch", "seq"])
    for k, v in p.items():
        g.init(k, v)
    one, axis0 = g.ints(1), g.ints(0)
    shape = g.op("Shape", ["tokens"])
    batch = g.op("Gather", [shape, g.init(g.fresh("i"), np.array(0, I64))], axis=0)
    seq = g.op("Gather", [shape, g.init(g.fresh("i"), np.array(1, I64))], axis=0)
    pos = g.op("Range", [g.init(g.fresh("i"), np.array(0, I64)), seq, g.init(g.fresh("i"), np.array(1, I64))])
    h = g.op("Add", [g.op("Gather", ["wte", "tokens"], axis=0), g.op("Gather", ["wpe", pos], axis=0)])
    b1, s1 = g.op("Unsqueeze", [batch, axis0]), g.op("Unsqueeze", [seq, axis0])
    heads = g.op("Concat", [b1, s1, g.ints(HEADS, dh)], axis=0)
    merged = g.op("Concat", [b1, s1, g.ints(-1)], axis=0)
    # The causal mask: True above the diagonal.
    square = g.op("Concat", [s1, s1], axis=0)
    ones = g.op("ConstantOfShape", [square], value=np.array([1], F32))
    above = g.op("Cast", [g.op("Trilu", [ones, one], upper=1)], to=BOOL)
    neg = g.scalar(-np.inf)
    scale = g.scalar(1 / math.sqrt(dh))
    for i in range(LAYERS):
        a = g.op("LayerNormalization", [h, f"h{i}.ln1.g", f"h{i}.ln1.b"], axis=-1, epsilon=1e-5)
        qkv = g.op("Add", [g.op("MatMul", [a, f"h{i}.qkv.w"]), f"h{i}.qkv.b"])
        q, k, v = g.op("Split", [qkv, g.ints(DIM, DIM, DIM)], 3, axis=2)
        q, k, v = (g.op("Transpose", [g.op("Reshape", [t, heads])], perm=[0, 2, 1, 3]) for t in (q, k, v))
        s = g.op("Mul", [g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])]), scale])
        s = g.op("Where", [above, neg, s])
        o = g.op("MatMul", [g.op("Softmax", [s], axis=-1), v])
        o = g.op("Reshape", [g.op("Transpose", [o], perm=[0, 2, 1, 3]), merged])
        h = g.op("Add", [h, g.op("Add", [g.op("MatMul", [o, f"h{i}.proj.w"]), f"h{i}.proj.b"])])
        m = g.op("LayerNormalization", [h, f"h{i}.ln2.g", f"h{i}.ln2.b"], axis=-1, epsilon=1e-5)
        f = g.op("Add", [g.op("MatMul", [m, f"h{i}.fc.w"]), f"h{i}.fc.b"])
        # GELU as torch.onnx.export writes it before opset 20.
        f = g.op("Mul", [g.op("Mul", [f, g.op("Add", [g.op("Erf", [g.op("Div", [f, g.scalar(math.sqrt(2))])]), g.scalar(1)])]), g.scalar(0.5)])
        h = g.op("Add", [h, g.op("Add", [g.op("MatMul", [f, f"h{i}.out.w"]), f"h{i}.out.b"])])
    h = g.op("LayerNormalization", [h, "lnf.g", "lnf.b"], axis=-1, epsilon=1e-5)
    g.op("MatMul", [h, "head.w"], ["logits"])
    g.output("logits", F32, ["batch", "seq", VOCAB])
    return g, p, {"tokens": tokens}, lambda tokens: {"logits": transformer_ref(p, tokens)}


def transformer_ref(p, tokens):
    t = tokens.shape[1]
    dh = DIM // HEADS
    h = p["wte"][tokens] + p["wpe"][:t]
    mask = np.triu(np.ones((t, t), bool), 1)
    for i in range(LAYERS):
        a = layer_norm(h, p[f"h{i}.ln1.g"], p[f"h{i}.ln1.b"])
        q, k, v = np.split(a @ p[f"h{i}.qkv.w"] + p[f"h{i}.qkv.b"], 3, axis=-1)
        q, k, v = (x.reshape(x.shape[0], t, HEADS, dh).transpose(0, 2, 1, 3) for x in (q, k, v))
        s = np.where(mask, -np.inf, q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh))
        o = (softmax(s) @ v).transpose(0, 2, 1, 3).reshape(h.shape)
        h = h + o @ p[f"h{i}.proj.w"] + p[f"h{i}.proj.b"]
        m = layer_norm(h, p[f"h{i}.ln2.g"], p[f"h{i}.ln2.b"])
        h = h + gelu(m @ p[f"h{i}.fc.w"] + p[f"h{i}.fc.b"]) @ p[f"h{i}.out.w"] + p[f"h{i}.out.b"]
    return layer_norm(h, p["lnf.g"], p["lnf.b"]) @ p["head.w"]


MODELS = {"mlp": mlp, "cnn": cnn, "decoder": decoder, "transformer": transformer}


# The same networks as PyTorch modules, for the torch oracle.


def torch_module(torch, name, p):
    nn, F = torch.nn, torch.nn.functional
    t = {k: torch.from_numpy(v) for k, v in p.items()}

    def param(x):
        return nn.Parameter(x.clone())

    if name == "mlp":
        m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10), nn.Softmax(-1))
        m[0].weight, m[0].bias, m[2].weight, m[2].bias = param(t["w1"]), param(t["b1"]), param(t["w2"]), param(t["b2"])
        return m
    if name == "cnn":
        m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(8, 16, 3, stride=2, padding=1, groups=2), nn.ReLU(), nn.AvgPool2d(2),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10), nn.Softmax(1))
        m[0].weight, m[0].bias = param(t["c1.w"]), param(t["c1.b"])
        m[1].weight, m[1].bias = param(t["bn.scale"]), param(t["bn.bias"])
        m[1].running_mean, m[1].running_var = t["bn.mean"].clone(), t["bn.var"].clone()
        m[4].weight, m[4].bias = param(t["c2.w"]), param(t["c2.b"])
        m[9].weight, m[9].bias = param(t["fc.w"]), param(t["fc.b"])
        return m.eval()
    if name == "decoder":
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1)
                self.out = nn.Conv2d(3, 2, 3)
                self.up.weight, self.up.bias = param(t["up.w"]), param(t["up.b"])
                self.out.weight, self.out.bias = param(t["out.w"]), param(t["out.b"])

            def forward(self, x):
                h = F.interpolate(self.up(x), scale_factor=2, mode="nearest")
                h = torch.sigmoid(self.out(F.pad(h, (1, 1, 1, 1), mode="reflect")))
                return h, F.interpolate(x, size=(7, 7), mode="bilinear", align_corners=False)

        return Decoder()

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.ParameterDict({k.replace(".", "_"): param(v) for k, v in t.items()})

        def forward(self, tokens):
            w = lambda k: self.p[k.replace(".", "_")]  # noqa: E731
            b, n = tokens.shape
            dh = DIM // HEADS
            h = w("wte")[tokens] + w("wpe")[torch.arange(n)]
            mask = torch.ones(n, n, dtype=torch.bool).triu(1)
            for i in range(LAYERS):
                a = F.layer_norm(h, (DIM,), w(f"h{i}.ln1.g"), w(f"h{i}.ln1.b"))
                q, k, v = (a @ w(f"h{i}.qkv.w") + w(f"h{i}.qkv.b")).split(DIM, dim=-1)
                q, k, v = (x.reshape(b, n, HEADS, dh).transpose(1, 2) for x in (q, k, v))
                s = (q @ k.transpose(-2, -1) / math.sqrt(dh)).masked_fill(mask, float("-inf"))
                o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, n, DIM)
                h = h + o @ w(f"h{i}.proj.w") + w(f"h{i}.proj.b")
                m = F.layer_norm(h, (DIM,), w(f"h{i}.ln2.g"), w(f"h{i}.ln2.b"))
                h = h + F.gelu(m @ w(f"h{i}.fc.w") + w(f"h{i}.fc.b")) @ w(f"h{i}.out.w") + w(f"h{i}.out.b")
            return F.layer_norm(h, (DIM,), w("lnf.g"), w("lnf.b")) @ w("head.w")

    return Transformer()


def export_torch(name, p, inputs, path):
    import torch

    torch.manual_seed(0)
    m = torch_module(torch, name, p).eval()
    args = tuple(torch.from_numpy(a) for a in inputs.values())
    names = {"mlp": ["probs"], "cnn": ["probs"], "decoder": ["image", "smooth"], "transformer": ["logits"]}[name]
    dynamic = {k: {0: "batch"} for k in inputs}
    if name == "transformer":
        dynamic = {"tokens": {0: "batch", 1: "seq"}}
    with torch.no_grad():
        torch.onnx.export(m, args, path, input_names=list(inputs), output_names=names, dynamic_axes=dynamic,
                          opset_version=17, dynamo=False)
        outs = m(*args)
    if not isinstance(outs, tuple):
        outs = (outs,)
    return {k: o.numpy().astype(F32) for k, o in zip(names, outs)}


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name in sorted(tensors):
        a = np.asarray(tensors[name], order="C")
        dtype = {np.dtype(F32): "F32", np.dtype(I32): "I32"}[a.dtype]
        data = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for data in blobs:
            f.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", choices=["torch", "numpy"], default="numpy")
    ap.add_argument("--out", default=os.path.join("io", "onnx", "testdata"))
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed + 12)
    for name, make in MODELS.items():
        g, p, inputs, ref = make(rng)
        path = os.path.join(args.out, name + ".onnx")
        if args.oracle == "torch":
            outs = export_torch(name, p, inputs, path)
        else:
            with open(path, "wb") as f:
                f.write(g.model())
            wide = {k: v.astype(np.float64) if v.dtype == F32 else v for k, v in inputs.items()}
            outs = {k: np.asarray(v).astype(F32) for k, v in ref(*wide.values()).items()}
        tensors = {f"in/{k}": v.astype(I32) if v.dtype == I64 else v for k, v in inputs.items()}
        tensors.update({f"out/{k}": v for k, v in outs.items()})
        write_safetensors(os.path.join(args.out, name + ".safetensors"), tensors, {"oracle": args.oracle})


if __name__ == "__main__":
    main()
