#!/usr/bin/env python3
"""Verify the clean Direct-44 release bank before serving it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
writer_dir = ROOT / "artifacts" / "gemma4-12b" / "writer"
bank = np.load(writer_dir / "direct44_raw_three_layer_writer.npz", allow_pickle=False)
report = json.loads((writer_dir / "writer_manifest.json").read_text(encoding="utf-8"))

assert len(bank["axes"]) == 44
assert len(set(bank["axes"].tolist())) == 44
assert bank["selected_layers"].shape == (44, 3)
assert bank["directions"].shape == (44, 3, 3840)
assert np.allclose(bank["layer_weights"], 0.6)
assert set(report["method"]["explicitly_excluded"]) >= {
    "legacy single-layer vectors",
    "Hybrid/Core/Natural blends",
    "subject/owner overlays",
    "anti-assistant overlays",
    "writer orthogonalization",
}
print("Gemma 4 12B bundle verified: 44 axes, 3 layers/axis, hidden=3840, gain=0.6/0.6/0.6")
