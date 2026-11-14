from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_release(size: str, hidden: int, layers: int) -> tuple[list[str], np.ndarray]:
    base = ROOT / "artifacts" / size
    writer = np.load(base / "writer" / "direct44_raw_three_layer_writer.npz", allow_pickle=False)
    reader = np.load(base / "reader" / "direct44_postwriter_live_affect_reader.npz", allow_pickle=False)

    axes = [str(value) for value in writer["axes"].tolist()]
    assert len(axes) == 44
    assert len(set(axes)) == 44
    assert writer["directions"].shape == (44, 3, hidden)
    assert writer["selected_layers"].shape == (44, 3)
    assert writer["layer_weights"].shape == (44, 3)
    assert np.allclose(writer["layer_weights"], 0.6)
    assert np.isfinite(writer["directions"]).all()
    assert int(writer["selected_layers"].min()) >= 0
    assert int(writer["selected_layers"].max()) < layers

    reader_axes = [str(value) for value in reader["axes"].tolist()]
    assert reader_axes == axes
    assert reader["directions"].shape == (44, hidden)
    assert reader["reader_layers"].shape == (44,)
    assert np.isfinite(reader["directions"]).all()
    assert np.all(reader["endpoint_half_spans"] > 0)
    assert int(reader["reader_layers"].min()) >= 0
    assert int(reader["reader_layers"].max()) < layers
    return axes, writer["categories"]


def test_12b_and_31b_artifacts_share_axis_contract() -> None:
    axes_12b, categories_12b = load_release("gemma4-12b", 3840, 48)
    axes_31b, categories_31b = load_release("gemma4-31b", 5376, 60)
    assert axes_12b == axes_31b
    assert categories_12b.tolist() == categories_31b.tolist()


def test_public_state_space_matches_artifacts() -> None:
    state_space = json.loads((ROOT / "configs" / "state_space.json").read_text())
    axes, _ = load_release("gemma4-31b", 5376, 60)
    assert [item["name"] for item in state_space["axes"]] == axes
