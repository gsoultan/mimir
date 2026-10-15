"""Times one training step in PyTorch for the models of mimir's
BenchmarkTrainStep: the forward pass, the loss, the backward pass and an
AdamW update.

    python3 train_step.py mlp|cnn|resnet18 [threads]
"""
import sys
import time

import torch
import torch.nn as nn

torch.manual_seed(1)
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10))


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + (x if self.down is None else self.down(x)))


def resnet18():
    layers = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
    cin = 64
    for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
        cin = cout
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1000)]
    return nn.Sequential(*layers)


name = sys.argv[1]
if name == "resnet18":
    batch, x, classes, warm, steps = 16, torch.rand(16, 3, 224, 224), 1000, 2, 5
else:
    batch, x, classes, warm, steps = 128, torch.rand(128, 1, 28, 28), 10, 20, 200
model = {"mlp": mlp, "cnn": cnn, "resnet18": resnet18}[name]()
y = torch.arange(batch) % classes
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()


def step():
    loss = lossf(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()


for _ in range(warm):
    step()
times = []
for _ in range(steps):
    t = time.perf_counter()
    step()
    times.append(time.perf_counter() - t)
times.sort()
print(f"{name}: torch {torch.__version__}, {torch.get_num_threads()} threads, "
      f"median {times[len(times) // 2] * 1e3:.2f} ms, best {times[0] * 1e3:.2f} ms")
