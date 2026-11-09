"""Trains the small CNN the web demo runs, and exports its weights for the browser.

    python training/train.py            # downloads MNIST (~11 MB), trains on CPU in a few minutes
    -> web/model/weights.json            # read by web/model.js

Architecture (kept small so every activation can be shown on screen):
    conv 3x3, 8 filters -> ReLU -> max-pool 2   (28x28x1 -> 14x14x8)
    conv 3x3, 16 filters -> ReLU -> max-pool 2  (14x14x8 -> 7x7x16)
    dense 64 -> ReLU -> dense 10
Training uses random shifts, rotations and scaling, because hand-drawn digits on a canvas are
less tidy than MNIST.
"""

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data"
OUT = ROOT / "web" / "model" / "weights.json"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
SEED = 0


def load(name: str) -> np.ndarray:
    path = DATA / FILES[name]
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR + FILES[name], path)
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(magic & 0xFF)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random rotation (±15°), scale (0.85-1.15) and shift (±3 px)."""
    n = x.shape[0]
    angle = (torch.rand(n) - 0.5) * (torch.pi / 6)
    scale = 0.85 + torch.rand(n) * 0.3
    shift = (torch.rand(n, 2) - 0.5) * (6 / 14)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1) / 255.0
    train_x, train_y = to_tensor(load("train_x")), torch.tensor(load("train_y"), dtype=torch.long)
    test_x, test_y = to_tensor(load("test_x")), torch.tensor(load("test_y"), dtype=torch.long)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    epochs = 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=epochs * (len(train_x) // 128 + 1))
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 128):
            idx = perm[i : i + 128]
            loss = F.cross_entropy(model(augment(train_x[idx])), train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(test_x).argmax(1) == test_y).float().mean().item()
        print(f"epoch {epoch + 1}: test accuracy {acc:.4f}", flush=True)

    # Export in TensorFlow.js layout: conv kernels [h, w, in, out]; the dense layer after the
    # convolutions expects activations flattened as (h, w, c), not PyTorch's (c, h, w).
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    fc1 = sd["fc1.weight"].reshape(64, 16, 7, 7).transpose(2, 3, 1, 0).reshape(7 * 7 * 16, 64)
    weights = {
        "conv1": {"kernel": sd["conv1.weight"].transpose(2, 3, 1, 0), "bias": sd["conv1.bias"]},
        "conv2": {"kernel": sd["conv2.weight"].transpose(2, 3, 1, 0), "bias": sd["conv2.bias"]},
        "fc1": {"kernel": fc1, "bias": sd["fc1.bias"]},
        "fc2": {"kernel": sd["fc2.weight"].T, "bias": sd["fc2.bias"]},
    }
    out = {
        layer: {k: {"shape": list(v.shape), "data": np.round(v.flatten(), 5).tolist()} for k, v in params.items()}
        for layer, params in weights.items()
    }
    out["meta"] = {"test_accuracy": round(acc, 4), "epochs": epochs, "seed": SEED}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), test accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
