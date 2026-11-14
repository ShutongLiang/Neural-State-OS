#!/usr/bin/env python3
"""Fast integrity check for the public Gemma 4 31B Direct-44 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WRITER_DIR = ROOT / "artifacts" / "gemma4-31b" / "writer"
WRITER_PATH = WRITER_DIR / "direct44_raw_three_layer_writer.npz"
MANIFEST_PATH = WRITER_DIR / "writer_manifest.json"


def main() -> None:
    writer = np.load(WRITER_PATH, allow_pickle=False)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    axes = [str(axis) for axis in writer["axes"].tolist()]
    directions = writer["directions"]
    selected_layers = writer["selected_layers"]
    weights = writer["layer_weights"]
    hidden_size = 5376
    num_layers = 60

    assert len(axes) == 44 and len(set(axes)) == 44
    assert directions.shape == (44, 3, hidden_size), directions.shape
    assert selected_layers.shape == (44, 3), selected_layers.shape
    assert weights.shape == (44, 3), weights.shape
    assert np.allclose(weights, 0.6), weights
    assert np.isfinite(directions).all()
    assert int(selected_layers.min()) >= 0
    assert int(selected_layers.max()) < num_layers
    assert [item["axis"] for item in manifest["axes"]] == axes
    assert manifest["method"]["scope"] == "all_tokens"

    print(
        "31B Direct-44 bundle verified: "
        f"44 axes, 3 layers/axis, hidden={hidden_size}, "
        f"layers=0..{num_layers - 1}, gain=0.6/0.6/0.6"
    )


if __name__ == "__main__":
    main()
