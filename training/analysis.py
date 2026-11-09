"""Offline interpretability analysis for the demo. Run after train.py.

    python training/analysis.py   ->  web/model/insights.json

1. Feature visualisation: for every filter and hidden unit, the input that excites it most
   (gradient ascent from a blank image, with jitter and an L2 penalty to keep it readable).
2. Dataset examples: the 4 MNIST test digits that activate each unit most.
3. Linear probes: for each layer (input, conv 1, conv 2, hidden), linear models trained on
   frozen activations to read out the digit and three concepts the network was never told about:
   whether the digit has a closed loop, its slant and its stroke thickness.
"""

import base64
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train import ROOT, Net, load

WEIGHTS = ROOT / "web" / "model" / "weights.json"
OUT = ROOT / "web" / "model" / "insights.json"
torch.manual_seed(0)
np.random.seed(0)


def load_model() -> Net:
    """Rebuilds the PyTorch model from the exported browser weights (inverse of train.py's export)."""
    raw = json.loads(WEIGHTS.read_text())
    arr = lambda layer, k: np.array(raw[layer][k]["data"], dtype=np.float32).reshape(raw[layer][k]["shape"])
    net = Net()
    sd = {
        "conv1.weight": arr("conv1", "kernel").transpose(3, 2, 0, 1),
        "conv1.bias": arr("conv1", "bias"),
        "conv2.weight": arr("conv2", "kernel").transpose(3, 2, 0, 1),
        "conv2.bias": arr("conv2", "bias"),
        "fc1.weight": arr("fc1", "kernel").reshape(7, 7, 16, 64).transpose(3, 2, 0, 1).reshape(64, 784),
        "fc1.bias": arr("fc1", "bias"),
        "fc2.weight": arr("fc2", "kernel").T,
        "fc2.bias": arr("fc2", "bias"),
    }
    net.load_state_dict({k: torch.tensor(v.copy()) for k, v in sd.items()})
    return net.eval()


def layers(net: Net, x: torch.Tensor) -> dict:
    """Activations used everywhere: conv outputs before pooling (NCHW) and the pooled, flattened
    features in the browser's (h, w, c) order."""
    a1 = F.relu(net.conv1(x))
    p1 = F.max_pool2d(a1, 2)
    a2 = F.relu(net.conv2(p1))
    p2 = F.max_pool2d(a2, 2)
    hidden = F.relu(net.fc1(p2.flatten(1)))
    hwc = lambda t: t.permute(0, 2, 3, 1).flatten(1)
    return {"a1": a1, "a2": a2, "hidden": hidden, "feat_input": x.flatten(1), "feat_conv1": hwc(p1), "feat_conv2": hwc(p2), "feat_hidden": hidden}


def b64(img: np.ndarray) -> str:
    return base64.b64encode(np.clip(img * 255, 0, 255).astype(np.uint8).tobytes()).decode()


# ---------- 1. feature visualisation ----------

def unit_value(net, x, layer, i):
    acts = layers(net, x)
    if layer == "hidden":
        return acts["hidden"][:, i]
    a = acts["a1" if layer == "conv1" else "a2"][:, i]
    return a.mean((1, 2))


def visualise(net, layer, i, steps=300):
    x = (torch.rand(1, 1, 28, 28) * 0.1).requires_grad_()
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(steps):
        dx, dy = np.random.randint(-2, 3, 2)
        jittered = torch.roll(x, shifts=(int(dx), int(dy)), dims=(2, 3))
        loss = -unit_value(net, jittered, layer, i).mean() + 0.02 * (x**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    img = x.detach()[0, 0].numpy()
    return (img - img.min()) / (img.max() - img.min() + 1e-9)


# ---------- 3. concepts ----------

def has_loop(img: np.ndarray) -> float:
    """1 if the ink encloses a background region of at least 4 pixels."""
    ink = img > 0.35
    seen = np.zeros_like(ink)
    q = deque((y, x) for y in range(28) for x in range(28) if (y in (0, 27) or x in (0, 27)) and not ink[y, x])
    for y, x in q:
        seen[y, x] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < 28 and 0 <= nx < 28 and not ink[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return float((~ink & ~seen).sum() >= 4)


def slant(img: np.ndarray) -> float:
    """Horizontal shift per row going up: positive leans right."""
    ys, xs = np.mgrid[0:28, 0:28]
    m = img.sum() + 1e-9
    my, mx = (img * ys).sum() / m, (img * xs).sum() / m
    cov = (img * (xs - mx) * (ys - my)).sum() / m
    var = (img * (ys - my) ** 2).sum() / m + 1e-9
    return float(-cov / var)


def thickness(img: np.ndarray) -> float:
    """Average stroke width in pixels: 2 x area / perimeter of the inked region."""
    ink = img > 0.35
    padded = np.pad(ink, 1)
    edge = ink & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return float(2 * ink.sum() / max(edge.sum(), 1))


def train_probe(xtr, ytr, xte, yte, kind):
    """Logistic (digit, loop) or linear (slant, thickness) probe on standardised features.
    Standardisation is folded into the weights so the browser only needs a dot product."""
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    ztr, zte = (xtr - mean) / std, (xte - mean) / std
    if kind in ("slant", "thickness"):
        return ridge_probe(ztr, ytr, zte, yte, mean, std)
    out = 10 if kind == "digit" else 1
    lin = torch.nn.Linear(ztr.shape[1], out)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        pred = lin(ztr)
        if kind == "digit":
            loss = F.cross_entropy(pred, ytr.long())
        elif kind == "loop":
            loss = F.binary_cross_entropy_with_logits(pred[:, 0], ytr)
        else:
            loss = F.mse_loss(pred[:, 0], ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = lin(zte)
        if kind == "digit":
            score = (pred.argmax(1) == yte.long()).float().mean().item()
        elif kind == "loop":
            score = ((pred[:, 0] > 0).float() == yte).float().mean().item()
        else:
            score = 1 - F.mse_loss(pred[:, 0], yte).item() / yte.var().item()  # R²
        w = (lin.weight / std).T  # [features, out]
        b = lin.bias - (lin.weight * (mean / std)).sum(1)
    return score, w.numpy(), b.numpy()


def ridge_probe(ztr, ytr, zte, yte, mean, std):
    """Closed-form ridge regression; the penalty is chosen on a held-out slice of the training set."""
    ztr, ytr, zte, yte = ztr.double(), ytr.double(), zte.double(), yte.double()
    n_fit = int(len(ztr) * 0.8)

    def fit(z, y, lam):
        a = z.T @ z + lam * torch.eye(z.shape[1], dtype=z.dtype)
        return torch.linalg.solve(a, z.T @ (y - y.mean())), y.mean()

    r2 = lambda pred, y: (1 - ((pred - y) ** 2).mean() / y.var()).item()

    def val_score(lam):
        w, b0 = fit(ztr[:n_fit], ytr[:n_fit], lam)
        return r2(ztr[n_fit:] @ w + b0, ytr[n_fit:])

    best = max((10.0**k for k in range(-1, 5)), key=val_score)
    w, b0 = fit(ztr, ytr, best)
    score = r2(zte @ w + b0, yte)
    w_raw = (w / std.double()).float()
    b = (b0 - (w * (mean.double() / std.double())).sum()).float()
    return score, w_raw.unsqueeze(1).numpy(), b.reshape(1).numpy()


def main():
    net = load_model()
    for p in net.parameters():
        p.requires_grad_(False)

    print("feature visualisation…", flush=True)
    featvis = {
        layer: [b64(visualise(net, layer, i)) for i in range(n)]
        for layer, n in (("conv1", 8), ("conv2", 16), ("hidden", 64))
    }

    test_x = torch.tensor(load("test_x"), dtype=torch.float32).unsqueeze(1) / 255
    print("dataset examples…", flush=True)
    with torch.no_grad():
        acts = layers(net, test_x)
    scores = {
        "conv1": acts["a1"].amax((2, 3)),
        "conv2": acts["a2"].amax((2, 3)),
        "hidden": acts["hidden"],
    }
    examples = {
        layer: [[b64(test_x[j, 0].numpy()) for j in s[:, i].topk(4).indices.tolist()] for i in range(s.shape[1])]
        for layer, s in scores.items()
    }

    print("probes…", flush=True)
    train_x = torch.tensor(load("train_x")[:12000], dtype=torch.float32).unsqueeze(1) / 255
    train_y = torch.tensor(load("train_y")[:12000], dtype=torch.float32)
    tx, ty = test_x[:3000], torch.tensor(load("test_y")[:3000], dtype=torch.float32)
    concept_fns = {"loop": has_loop, "slant": slant, "thickness": thickness}
    concepts = {
        split: {name: torch.tensor([fn(img[0].numpy()) for img in xs], dtype=torch.float32) for name, fn in concept_fns.items()}
        for split, xs in (("train", train_x), ("test", tx))
    }
    with torch.no_grad():
        ftr, fte = layers(net, train_x), layers(net, tx)
    layer_names = ["input", "conv1", "conv2", "hidden"]
    probes = {"layers": layer_names, "scores": {}, "weights": {}}
    for name in ["digit", "loop", "slant", "thickness"]:
        probes["scores"][name] = []
        for layer in layer_names:
            ytr = train_y if name == "digit" else concepts["train"][name]
            yte = ty if name == "digit" else concepts["test"][name]
            score, w, b = train_probe(ftr[f"feat_{layer}"], ytr, fte[f"feat_{layer}"], yte, name)
            probes["scores"][name].append(round(score, 4))
            probes["weights"].setdefault(layer, {})[name] = {
                "shape": list(w.shape),
                "w": np.round(w.flatten(), 4).tolist(),
                "b": np.round(b, 4).tolist(),
            }
            print(f"  {name:9s} {layer:6s} {score:.3f}", flush=True)
    probes["ranges"] = {k: [round(float(v.min()), 3), round(float(v.max()), 3)] for k, v in concepts["train"].items()}

    OUT.write_text(json.dumps({"featvis": featvis, "examples": examples, "probes": probes}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
