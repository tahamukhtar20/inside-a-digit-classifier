"""Exports 100 MNIST test digits (10 per class) for the demo's "Random MNIST digit" button."""

import json
from pathlib import Path

import numpy as np

from train import load

OUT = Path(__file__).resolve().parent.parent / "web" / "model" / "samples.json"

x, y = load("test_x"), load("test_y")
rng = np.random.default_rng(0)
picks = [i for d in range(10) for i in rng.choice(np.flatnonzero(y == d), 10, replace=False)]
OUT.write_text(json.dumps([{"label": int(y[i]), "pixels": x[i].flatten().tolist()} for i in picks], separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
