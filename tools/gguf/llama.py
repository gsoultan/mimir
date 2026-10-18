#!/usr/bin/env python3
"""Generate the tiny Llama models that examples/llama's tests read.

    pip install gguf numpy
    python tools/gguf/llama.py --out /tmp/llama
    for m in spm bpe; do
        go run github.com/gsoultan/mim/cmd/mim-import -o examples/llama/testdata/$m.mim /tmp/llama/$m.gguf
        go run github.com/gsoultan/mim/cmd/mim-import -o examples/llama/testdata/$m.logits.mim /tmp/llama/$m.logits.safetensors
    done

The tests read Mimir Model files, which mim-import converts the script's
output to, copying the tensors byte for byte and the GGUF metadata (the
hyperparameters and the tokenizer) as it is.

It writes two models with random weights, each a GGUF file that llama.cpp
loads, and the logits of a few tokens that a NumPy implementation of the
model computes in float64, as <name>.logits.safetensors:

    spm.gguf  two blocks of 64 features in 4 heads of 16 with 2 key and
              value heads, an MLP of 128, rope base 10000, linear rope
              scaling by 2, an output projection of its own, weights in
              q8_0, f16, bf16 and f32, and a SentencePiece ("llama")
              tokenizer of 256 byte tokens, a few pieces and two special
              tokens

    bpe.gguf  one block of 256 features in 4 heads of 64 with one key and
              value head, an MLP of 256, rope base 500000 with Llama 3.1's
              frequency factors in rope_freqs.weight, the output tied to
              the token embedding, weights in q4_K and q6_K, and a GPT-2
              byte-level BPE ("gpt2") tokenizer with Llama 3's
              pre-tokenizer

The k-quant weights are random blocks with small scales, so that their
values are those of a trained model's in size; gguf-py quantizes the rest
from random floats, and dequantizes every weight for the reference.
Rotary embeddings rotate interleaved pairs of features, as llama.cpp
rotates the "llama" architecture's.
"""

import argparse
import json
import os
import struct

import numpy as np

import gguf
from gguf import GGMLQuantizationType as T


def kquant(rng, qtype, rows, cols):
    """Return random blocks of a k-quant type of rows×cols weights."""
    size = gguf.GGML_QUANT_SIZES[qtype][1]
    blocks = rows * cols // 256
    b = rng.integers(0, 256, size=(blocks, size), dtype=np.uint8)
    if qtype == T.Q4_K:
        # d·scale·q - dmin·min with 6-bit scale and min and 4-bit q, whose
        # mean is about 0 when dmin is 7.5·d.
        b[:, 0:2] = np.full(blocks, 1 / 4096, np.float16).view(np.uint8).reshape(-1, 2)
        b[:, 2:4] = np.full(blocks, 7.5 / 4096, np.float16).view(np.uint8).reshape(-1, 2)
    elif qtype == T.Q6_K:
        # d·scale·(q-32) with an int8 scale and 6-bit q.
        b[:, size - 2 :] = np.full(blocks, 1 / 16384, np.float16).view(np.uint8).reshape(-1, 2)
    else:
        raise ValueError(qtype)
    return b.reshape(rows, -1)


def weight(rng, qtype, rows, cols, std):
    """Return a weight of shape (rows, cols) as stored and dequantized."""
    if qtype in (T.Q4_K, T.Q6_K):
        data = kquant(rng, qtype, rows, cols)
    else:
        x = (rng.standard_normal((rows, cols)) * std).astype(np.float32)
        if qtype == T.F32:
            data = x
        elif qtype == T.F16:
            data = x.astype(np.float16)
        else:
            data = gguf.quants.quantize(x, qtype)
    want = gguf.quants.dequantize(data, qtype) if qtype not in (T.F32, T.F16) else data
    return data, np.asarray(want, np.float64).reshape(rows, cols)


def norm_weight(rng, n):
    return (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos, base, freq_scale, factors):
    """Rotate interleaved pairs of x, [S, heads, dim], at positions pos."""
    d = x.shape[-1]
    inv = base ** (-np.arange(0, d, 2) / d) * freq_scale / factors
    ang = pos[:, None, None] * inv[None, None, :]
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def forward(m, tokens):
    """Return the logits, [S, vocab], of the model m for tokens."""
    h = m["token_embd.weight"][tokens]
    s = len(tokens)
    pos = np.arange(s, dtype=np.float64)
    heads, kv, dim = m["heads"], m["kv_heads"], m["head_dim"]
    mask = np.triu(np.full((s, s), -np.inf), 1)
    for i in range(m["layers"]):
        p = f"blk.{i}."
        x = rms_norm(h, m[p + "attn_norm.weight"], m["eps"])
        q = (x @ m[p + "attn_q.weight"].T).reshape(s, heads, dim)
        k = (x @ m[p + "attn_k.weight"].T).reshape(s, kv, dim)
        v = (x @ m[p + "attn_v.weight"].T).reshape(s, kv, dim)
        q = rope(q, pos, m["rope_base"], m["freq_scale"], m["factors"])
        k = rope(k, pos, m["rope_base"], m["freq_scale"], m["factors"])
        group = heads // kv
        out = np.empty((s, heads, dim))
        for hd in range(heads):
            j = hd // group
            a = q[:, hd] @ k[:, j].T / np.sqrt(dim) + mask
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            out[:, hd] = a @ v[:, j]
        h = h + out.reshape(s, heads * dim) @ m[p + "attn_output.weight"].T
        x = rms_norm(h, m[p + "ffn_norm.weight"], m["eps"])
        g = x @ m[p + "ffn_gate.weight"].T
        u = x @ m[p + "ffn_up.weight"].T
        h = h + (g / (1 + np.exp(-g)) * u) @ m[p + "ffn_down.weight"].T
    h = rms_norm(h, m["output_norm.weight"], m["eps"])
    out = m.get("output.weight", m["token_embd.weight"])
    return h @ out.T


def llama3_factors(dim, base, factor, low, high, original):
    """Return Llama 3.1's divisors of each pair's frequency."""
    inv = base ** (-np.arange(0, dim, 2) / dim)
    wave = 2 * np.pi / inv
    low_wave, high_wave = original / low, original / high
    out = np.ones(dim // 2)
    for i, w in enumerate(wave):
        if w < high_wave:
            out[i] = 1
        elif w > low_wave:
            out[i] = factor
        else:
            smooth = (original / w - low) / (high - low)
            out[i] = 1 / ((1 - smooth) / factor + smooth)
    return out


# GPT-2's map of bytes to printable characters.
def byte_chars():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


SPM_PIECES = [
    "▁", "e", "t", "a", "o", "i", "n", "s", "h", "r", "l", "d", "u", "w", "m", "c", "g", "y", "f", "p",
    "b", "k", "v", "H", "T", "W", ",", ".", "!", "'", "1", "2", "3",
    "▁t", "he", "▁the", "in", "er", "▁a", "on", "▁w", "or", "ll", "▁H", "▁He", "▁Hell", "▁Hello",
    "▁wor", "▁world", "ld", "is", "▁is", "▁th", "▁this", "at", "en", "▁in", "an", "it", "▁it",
    "es", "ing", "▁wh", "▁what", "▁What", "▁W", "▁T", "▁The",
]


def spm_vocab():
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + SPM_PIECES + ["<|user|>", "<|end|>"]
    types = [2, 3, 3] + [6] * 256 + [1] * len(SPM_PIECES) + [4, 3]
    # Longer pieces score higher, as SentencePiece's do; ties go to the
    # piece listed first.
    scores = [0.0] * 259 + [-1000.0 + 10 * len(p) - i * 0.01 for i, p in enumerate(SPM_PIECES)] + [0.0, 0.0]
    return tokens, types, scores


BPE_MERGES = [
    "Ġ t", "h e", "Ġt he", "Ġ a", "i n", "e r", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld",
    "H e", "l l", "He ll", "Hell o", "Ġ H", "ĠH ello", "e llo", "i s", "Ġ is", "Ġt h", "Ġth is",
    "1 2", "12 3", "Ċ Ċ", "Ġ Ġ", "' s", "a t", "Ġ the", "o n",
]


def bpe_vocab():
    chars = byte_chars()
    tokens = [chars[b] for b in range(256)]
    for m in BPE_MERGES:
        t = m.replace(" ", "")
        if t not in tokens:
            tokens.append(t)
    tokens += ["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|start_header_id|>"]
    types = [1] * (len(tokens) - 4) + [3, 3, 3, 3]
    return tokens, types


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, a in tensors.items():
        b = a.tobytes()
        dtype = {np.dtype(np.float32): "F32", np.dtype(np.int32): "I32"}[a.dtype]
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def build(path, name, rng, cfg, types, vocab_kv, prompt):
    embed, heads, kv, hidden = cfg["embed"], cfg["heads"], cfg["kv_heads"], cfg["hidden"]
    dim = embed // heads
    w = gguf.GGUFWriter(path, "llama")
    w.add_name(name)
    w.add_context_length(cfg["context"])
    w.add_embedding_length(embed)
    w.add_block_count(cfg["layers"])
    w.add_feed_forward_length(hidden)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_rope_dimension_count(dim)
    w.add_rope_freq_base(cfg["rope_base"])
    w.add_layer_norm_rms_eps(cfg["eps"])
    w.add_vocab_size(cfg["vocab"])
    if cfg.get("linear_scaling"):
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        w.add_rope_scaling_factor(cfg["linear_scaling"])
    vocab_kv(w)

    m = dict(heads=heads, kv_heads=kv, head_dim=dim, layers=cfg["layers"], eps=cfg["eps"],
             rope_base=cfg["rope_base"], freq_scale=1 / cfg.get("linear_scaling", 1), factors=np.ones(dim // 2))

    def add(tname, qtype, rows, cols, std=0.05):
        data, want = weight(rng, qtype, rows, cols, std)
        w.add_tensor(tname, data, raw_dtype=None if qtype in (T.F32, T.F16) else qtype)
        m[tname] = want

    def add_norm(tname, n):
        x = norm_weight(rng, n)
        w.add_tensor(tname, x)
        m[tname] = x.astype(np.float64)

    add("token_embd.weight", types["embd"], cfg["vocab"], embed, 1.0)
    if "factors" in cfg:
        f = cfg["factors"].astype(np.float32)
        w.add_tensor("rope_freqs.weight", f)
        m["factors"] = f.astype(np.float64)
    for i in range(cfg["layers"]):
        p = f"blk.{i}."
        add_norm(p + "attn_norm.weight", embed)
        add(p + "attn_q.weight", types["q"], heads * dim, embed)
        add(p + "attn_k.weight", types["k"], kv * dim, embed)
        add(p + "attn_v.weight", types["v"], kv * dim, embed)
        add(p + "attn_output.weight", types["o"], embed, heads * dim)
        add_norm(p + "ffn_norm.weight", embed)
        add(p + "ffn_gate.weight", types["gate"], hidden, embed)
        add(p + "ffn_up.weight", types["up"], hidden, embed)
        add(p + "ffn_down.weight", types["down"], embed, hidden)
    add_norm("output_norm.weight", embed)
    if "out" in types:
        add("output.weight", types["out"], cfg["vocab"], embed)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    tokens = np.array(prompt, np.int32)
    logits = forward(m, tokens).astype(np.float32)
    write_safetensors(path.removesuffix(".gguf") + ".logits.safetensors", {"tokens": tokens, "logits": logits})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokens, types, scores = spm_vocab()

    def spm_kv(w):
        w.add_tokenizer_model("llama")
        w.add_token_list(tokens)
        w.add_token_scores(scores)
        w.add_token_types(types)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_unk_token_id(0)
        w.add_add_bos_token(True)
        w.add_add_eos_token(False)

    spm = dict(context=64, embed=64, heads=4, kv_heads=2, hidden=128, layers=2, rope_base=10000.0, eps=1e-5,
               vocab=len(tokens), linear_scaling=2.0)
    spm_types = dict(embd=T.Q8_0, q=T.Q8_0, k=T.Q8_0, v=T.Q8_0, o=T.F16, gate=T.Q8_0, up=T.F32, down=T.BF16, out=T.Q8_0)
    build(os.path.join(args.out, "spm.gguf"), "tiny spm llama", np.random.default_rng(1), spm, spm_types, spm_kv,
          [1, 259 + SPM_PIECES.index("▁Hello"), 259 + SPM_PIECES.index("▁world"), 259 + SPM_PIECES.index("!"), 5, 300, 77, 2])

    btokens, btypes = bpe_vocab()

    def bpe_kv(w):
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("llama-bpe")
        w.add_token_list(btokens)
        w.add_token_types(btypes)
        w.add_token_merges(BPE_MERGES)
        w.add_bos_token_id(btokens.index("<|begin_of_text|>"))
        w.add_eos_token_id(btokens.index("<|end_of_text|>"))
        w.add_eot_token_id(btokens.index("<|eot_id|>"))
        w.add_add_bos_token(True)

    bpe = dict(context=128, embed=256, heads=4, kv_heads=1, hidden=256, layers=1, rope_base=500000.0, eps=1e-6,
               vocab=len(btokens), factors=llama3_factors(64, 500000.0, 8.0, 1.0, 4.0, 16))
    bpe_types = dict(embd=T.Q6_K, q=T.Q4_K, k=T.Q4_K, v=T.Q6_K, o=T.Q4_K, gate=T.Q4_K, up=T.Q4_K, down=T.Q6_K)
    build(os.path.join(args.out, "bpe.gguf"), "tiny bpe llama", np.random.default_rng(2), bpe, bpe_types, bpe_kv,
          [btokens.index("<|begin_of_text|>"), btokens.index("Hello"), btokens.index("Ġworld"), 0, 17, 260, 3, 255, 100, 200, 42, 7])


if __name__ == "__main__":
    main()
