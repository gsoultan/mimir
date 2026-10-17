#!/usr/bin/env python3
"""Generate the GGUF file that io/gguf's tests read.

    pip install gguf numpy
    python tools/gguf/gen.py --out io/gguf/testdata

It writes small.gguf with llama.cpp's own writer, gguf-py, so the tests
check the Go reader against the format as llama.cpp writes it rather
than against io/gguf's writer. The file aligns its tensor data to 64
bytes, not the default 32, and holds

    metadata  the keys of a llama model's header (general.architecture,
              llama.block_count, ...), a tokenizer's arrays (tokens,
              scores, token_type, merges), one key test.<type> of every
              scalar type and an array of arrays, test.nested

    tensors   random weights of each type mimir reads, q8_0, q4_K, q6_K,
              f16, bf16 and f32, of up to four dimensions, one q5_K
              weight, which mimir does not read, and for each quantized
              or half weight w a float32 tensor "w.want" that gguf-py
              dequantized it to

The quantized weights are random bytes whose float16 scales are drawn
from a few powers of two around 1/64, so that no scale is a NaN or an
infinity, except q8_0's, which gguf-py quantizes from random floats.
gguf-py dequantizes rounding every product as ggml does, so the tests
compare mimir's dequantization with the .want tensors bit for bit.
"""

import argparse
import os

import numpy as np

import gguf
from gguf import GGMLQuantizationType as T
from gguf import GGUFValueType as V


def scales(rng, n):
    """Return n float16 scales, as bytes, of either sign."""
    s = rng.choice([1 / 16, 1 / 32, 1 / 64, 1 / 128, 3 / 256], size=n)
    s = s * rng.choice([-1, 1], size=n)
    return s.astype(np.float16).view(np.uint8).reshape(n, 2)


def kquant(rng, qtype, blocks):
    """Return random blocks of a k-quant type as uint8 rows."""
    size = gguf.GGML_QUANT_SIZES[qtype][1]
    b = rng.integers(0, 256, size=(blocks, size), dtype=np.uint8)
    if qtype == T.Q4_K or qtype == T.Q5_K:
        b[:, 0:2] = scales(rng, blocks)  # d
        b[:, 2:4] = np.abs(scales(rng, blocks).view(np.float16)).view(np.uint8)  # dmin
    elif qtype == T.Q6_K:
        b[:, size - 2 :] = scales(rng, blocks)  # d, after ql, qh and scales
    else:
        raise ValueError(qtype)
    return b


def weight(rng, qtype, shape):
    """Return the bytes of a random weight of shape, as rows of uint8."""
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if qtype == T.F32:
        return rng.standard_normal(shape).astype(np.float32)
    if qtype == T.F16:
        return rng.standard_normal(shape).astype(np.float16)
    if qtype == T.BF16:
        x = rng.standard_normal(shape).astype(np.float32)
        return gguf.quants.quantize(x, T.BF16)
    if qtype == T.Q8_0:
        x = rng.standard_normal(shape).astype(np.float32)
        return gguf.quants.quantize(x, T.Q8_0)
    bs = gguf.GGML_QUANT_SIZES[qtype][0]
    b = kquant(rng, qtype, rows * cols // bs)
    return b.reshape(*shape[:-1], -1)


TENSORS = [
    ("token_embd.weight", T.Q6_K, (3, 256)),
    ("blk.0.attn_q.weight", T.Q4_K, (2, 512)),
    ("blk.0.ffn_down.weight", T.Q8_0, (3, 64)),
    ("blk.0.ffn_up.weight", T.Q8_0, (2, 1, 2, 32)),
    ("blk.0.attn_norm.weight", T.F32, (8,)),
    ("blk.0.attn_k.weight", T.F16, (2, 3)),
    ("blk.0.attn_v.weight", T.BF16, (5,)),
    ("output.weight", T.Q5_K, (1, 256)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    w = gguf.GGUFWriter(os.path.join(args.out, "small.gguf"), "llama")
    w.add_custom_alignment(64)
    w.add_name("small")
    w.add_block_count(1)
    w.add_context_length(128)
    w.add_embedding_length(256)
    w.add_head_count(4)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["<s>", "</s>", "a", "b", "ab", "été"])
    w.add_token_scores([0.0, 0.0, -1.0, -1.5, -2.0, -2.25])
    w.add_token_types([3, 3, 1, 1, 1, 1])
    w.add_token_merges(["a b"])
    w.add_key_value("test.uint8", 200, V.UINT8)
    w.add_key_value("test.int8", -100, V.INT8)
    w.add_key_value("test.uint16", 60000, V.UINT16)
    w.add_key_value("test.int16", -30000, V.INT16)
    w.add_key_value("test.uint32", 4000000000, V.UINT32)
    w.add_key_value("test.int32", -2000000000, V.INT32)
    w.add_key_value("test.float32", 0.25, V.FLOAT32)
    w.add_key_value("test.bool", True, V.BOOL)
    w.add_key_value("test.string", "héllo", V.STRING)
    w.add_key_value("test.uint64", 2**63 + 5, V.UINT64)
    w.add_key_value("test.int64", -(2**62), V.INT64)
    w.add_key_value("test.float64", -1.5e300, V.FLOAT64)
    w.add_key_value("test.nested", [[1, 2], [3]], V.ARRAY, sub_type=V.ARRAY)

    for name, qtype, shape in TENSORS:
        data = weight(rng, qtype, shape)
        if qtype in (T.F32, T.F16):
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=qtype)
        if qtype not in (T.F32, T.Q5_K):
            want = gguf.quants.dequantize(data, qtype).astype(np.float32)
            w.add_tensor(name + ".want", want.reshape(shape))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
