"""Times in PyTorch the work of mimir's attention benchmarks:

    attention  causal attention at GPT-2 small shapes (8 sequences, 12 heads
               of 64 features), forward and backward (BenchmarkAttention in
               package tensor), with F.scaled_dot_product_attention
    block      a training step of a GPT-2 small block: forward, a loss,
               backward and AdamW (BenchmarkDecoderBlockStep in package nn)
    decode     a decoding step of one token with 512 positions in the cache
               at Llama 3.2 1B shapes (BenchmarkDecode in package nn)
    lstm       a 2-layer LSTM of 256 hidden units over 128 steps of 32
               sequences of 64 features: forward, forward and backward, and
               a training step with AdamW (BenchmarkLSTM in package nn)

    python3 attention.py attention|block|decode|lstm [threads] [device]

device is cpu (the default) or cuda; on cuda each time waits for the GPU.
"""
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
torch.set_default_device(device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(name, f, warm, steps):
    for _ in range(warm):
        f()
    sync()
    times = []
    for _ in range(steps):
        t = time.perf_counter()
        f()
        sync()
        times.append(time.perf_counter() - t)
    times.sort()
    print(f"{name}: best {times[0] * 1e3:.2f} ms, median {times[len(times) // 2] * 1e3:.2f} ms")


def attention():
    for seq in (256, 1024):
        q, k, v = (torch.randn(8, 12, seq, 64, requires_grad=True) for _ in range(3))
        w = torch.randn(8, 12, seq, 64)

        def step():
            (F.scaled_dot_product_attention(q, k, v, is_causal=True) * w).sum().backward()
            for x in (q, k, v):
                x.grad = None

        timed(f"attention seq{seq}", step, 2, 5)


class GPT2Block(nn.Module):
    def __init__(self, embed, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(embed), nn.LayerNorm(embed)
        self.qkv, self.proj = nn.Linear(embed, 3 * embed), nn.Linear(embed, embed)
        self.up, self.down = nn.Linear(embed, 4 * embed), nn.Linear(4 * embed, embed)

    def forward(self, x):
        b, s, e = x.shape
        q, k, v = (t.unflatten(-1, (self.heads, e // self.heads)).transpose(1, 2)
                   for t in self.qkv(self.ln1(x)).split(e, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).flatten(-2))
        return x + self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh"))


def block():
    for seq in (256, 1024):
        model = GPT2Block(768, 12)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, w = torch.randn(8, seq, 768), torch.randn(8, seq, 768)

        def step():
            (model(x) * w).mean().backward()
            opt.step()
            opt.zero_grad()

        timed(f"block step seq{seq}", step, 2, 5)


def rope(x, pos, dim=64, base=500000.0):
    inv = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    ang = (pos * inv).float()
    cos, sin = ang.cos(), ang.sin()
    a, b = x[..., : dim // 2], x[..., dim // 2:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


class LlamaBlock(nn.Module):
    def __init__(self, embed=2048, heads=32, kv=8, hidden=8192):
        super().__init__()
        self.heads, self.kv, self.dim = heads, kv, embed // heads
        self.n1, self.n2 = nn.RMSNorm(embed, eps=1e-5), nn.RMSNorm(embed, eps=1e-5)
        self.qkv = nn.Linear(embed, (heads + 2 * kv) * self.dim, bias=False)
        self.o = nn.Linear(embed, embed, bias=False)
        self.gate, self.up = nn.Linear(embed, hidden, bias=False), nn.Linear(embed, hidden, bias=False)
        self.down = nn.Linear(hidden, embed, bias=False)

    def decode(self, x, cache_k, cache_v, past):
        b, s, e = x.shape
        d, h, hk = self.dim, self.heads, self.kv
        q, k, v = self.qkv(self.n1(x)).split([h * d, hk * d, hk * d], dim=-1)
        q = rope(q.unflatten(-1, (h, d)).transpose(1, 2), past)
        k = rope(k.unflatten(-1, (hk, d)).transpose(1, 2), past)
        cache_k[:, :, past:past + s] = k
        cache_v[:, :, past:past + s] = v.unflatten(-1, (hk, d)).transpose(1, 2)
        keys, values = cache_k[:, :, :past + s], cache_v[:, :, :past + s]
        a = F.scaled_dot_product_attention(q, keys, values, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).flatten(-2))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


@torch.no_grad()
def decode():
    past = 512
    cache_k, cache_v = torch.zeros(1, 8, past + 1, 64), torch.zeros(1, 8, past + 1, 64)
    cache_k[:, :, :past], cache_v[:, :, :past] = torch.randn(1, 8, past, 64), torch.randn(1, 8, past, 64)
    q, k, v = torch.randn(1, 32, 1, 64), torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64)

    def attend():
        cache_k[:, :, past:past + 1], cache_v[:, :, past:past + 1] = k, v
        F.scaled_dot_product_attention(q, cache_k, cache_v, enable_gqa=True)

    timed("decode attention", attend, 10, 100)
    model = LlamaBlock().eval()
    x = torch.randn(1, 1, 2048)
    timed("decode block", lambda: model.decode(x, cache_k, cache_v, past), 3, 20)


def lstm():
    steps, batch, inp, hidden = 128, 32, 64, 256
    model = nn.LSTM(inp, hidden, num_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.randn(steps, batch, inp), torch.randn(steps, batch, hidden)

    def forward():
        with torch.no_grad():
            model(x)

    def backward():
        F.mse_loss(model(x)[0], y).backward()
        model.zero_grad()

    def train():
        F.mse_loss(model(x)[0], y).backward()
        opt.step()
        opt.zero_grad()

    timed("lstm forward", forward, 3, 10)
    timed("lstm backward", backward, 3, 10)
    timed("lstm train", train, 3, 10)


{"attention": attention, "block": block, "decode": decode, "lstm": lstm}[sys.argv[1]]()
