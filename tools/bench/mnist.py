"""Trains examples/mnist's convolutional network in PyTorch with the same
settings: AdamW at 1e-3 with weight decay 0.01, cosine annealing, batches
of 128.

    python3 mnist.py DATA_DIR [epochs]

DATA_DIR holds the four MNIST .gz files, as examples/mnist downloads them.
"""
import gzip
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1)
torch.set_num_threads(4)
data = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def images(name):
    with gzip.open(f"{data}/{name}") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).astype(np.float32)
    return torch.from_numpy(x.reshape(-1, 1, 28, 28) / 255)


def labels(name):
    with gzip.open(f"{data}/{name}") as f:
        return torch.from_numpy(np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64))


xtr, ytr = images("train-images-idx3-ubyte.gz"), labels("train-labels-idx1-ubyte.gz")
xte, yte = images("t10k-images-idx3-ubyte.gz"), labels("t10k-labels-idx1-ubyte.gz")
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
lossf = nn.CrossEntropyLoss()
g = torch.Generator().manual_seed(42)
for epoch in range(epochs):
    t = time.perf_counter()
    model.train()
    for idx in torch.randperm(len(xtr), generator=g).split(128):
        loss = lossf(model(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    schedule.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(xte).argmax(1) == yte).float().mean().item()
    print(f"epoch {epoch + 1}: accuracy {accuracy:.4f} ({time.perf_counter() - t:.1f}s)", flush=True)
