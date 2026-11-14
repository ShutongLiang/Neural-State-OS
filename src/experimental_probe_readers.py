"""Experimental hidden-state probes for the Direct-44 demo.

These readers are deliberately separate from the 44-axis production-style
reader.  They compare two research hypotheses on six gate axes:

* Persona: a labelled high-minus-low activation direction;
* SAE: one sparse TopK feature discovered without labels, then labelled.

Both banks were calibrated only on Direct-44 internal-experience prompts.
They are observable research instruments, not a claim that a number is a
ground-truth emotion or a diagnosis of an agent's experience.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(
    os.environ.get(
        "NEURAL_STATE_OS_EXPERIMENT_ROOT",
        str(Path(__file__).resolve().parents[1] / "artifacts" / "experimental-probes"),
    )
)
PERSONA_BANK = Path(
    os.environ.get(
        "NEURAL_STATE_OS_PERSONA_BANK",
        str(
            EXPERIMENT_ROOT
            / "artifacts/persona_style_direct44/persona_style_prompt_tail_reader.npz"
        ),
    )
)
SAE_DIR = EXPERIMENT_ROOT / "artifacts/saelens_topk_centered300_direct44"
SAE_READER = SAE_DIR / "saelens_topk_feature_reader.npz"
SAE_OFFSETS = SAE_DIR / "saelens_topk_runtime_offsets.npz"


def _ranked(values: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(values, key=lambda axis: abs(float(values[axis]["value"])), reverse=True)


@dataclass(frozen=True)
class _PersonaAxis:
    layer: int
    direction: np.ndarray
    center: float
    half_span: float


class PersonaVectorProbe:
    name = "persona"

    def __init__(self, path: Path = PERSONA_BANK) -> None:
        bank = np.load(path, allow_pickle=False)
        axes = [str(axis) for axis in bank["axes"].tolist()]
        layers = bank["reader_layers"].astype(np.int32)
        directions = bank["directions"].astype(np.float32)
        centers = bank["centers"].astype(np.float32)
        spans = bank["half_spans"].astype(np.float32)
        daily_centers = (
            bank["daily_centers"].astype(np.float32)
            if "daily_centers" in bank.files
            else np.zeros(len(axes), dtype=np.float32)
        )
        if directions.ndim != 2 or directions.shape[0] != len(axes):
            raise ValueError("Persona experimental bank has incompatible directions")
        if not np.all(spans > 1e-6):
            raise ValueError("Persona experimental bank has invalid calibration")
        self.axes = axes
        self.axis_count = len(axes)
        self.daily_centered = "daily_centers" in bank.files
        if daily_centers.shape != (len(axes),):
            raise ValueError("Persona daily calibration dimensions disagree")
        self.daily_centers = {
            axis: float(value) for axis, value in zip(axes, daily_centers)
        }
        self.label = (
            "Persona state · three-gate validation"
            if self.daily_centered
            else "Persona direction · calibration-domain candidate"
        )
        self.note = (
            "Shows only axes that pass cross-scene readout, independent daily-chat "
            "false-positive checks, and causal writer-dose response. value is daily-"
            "baseline centered; raw_value retains the calibration-domain scale."
            if self.daily_centered
            else "Mean hidden-activation difference between positive and negative "
            "calibration records. Candidate axes require separate daily-chat false-"
            "positive and causal writer-response checks."
        )
        self.by_axis = {
            axis: _PersonaAxis(int(layer), direction, float(center), float(span))
            for axis, layer, direction, center, span in zip(
                axes, layers, directions, centers, spans
            )
        }
        self.layers = sorted({item.layer for item in self.by_axis.values()})

    def observe(self, activations: dict[int, np.ndarray]) -> dict[str, Any]:
        values: dict[str, dict[str, Any]] = {}
        for axis, item in self.by_axis.items():
            raw = float(np.dot(activations[item.layer], item.direction))
            raw_value = (raw - item.center) / item.half_span
            value = raw_value - self.daily_centers[axis]
            values[axis] = {
                "value": round(float(np.clip(value, -3.0, 3.0)), 4),
                "raw_value": round(float(raw_value), 4),
                "layer": item.layer,
            }
        return {
            "enabled": True,
            "label": self.label,
            "note": f"{self.note} Axis count: {self.axis_count}.",
            "values": values,
            "top_observed": _ranked(values)[:6],
            "scale": (
                "0 is the daily-chat baseline; ±1 is approximately half the "
                "calibration-domain endpoint separation."
                if self.daily_centered
                else "0 is the midpoint between calibration endpoints; ±1 is "
                "approximately half their separation."
            ),
        }


@dataclass(frozen=True)
class _SaeAxis:
    layer: int
    feature: int
    center: float
    half_span: float


class SaeTopKProbe:
    name = "sae"
    label = "SAE sparse feature · calibration-domain pilot"
    note = (
        "Learns sparse TopK features without labels, then selects one feature per axis. "
        "Six-axis held-scene intensity validation ranges from roughly 0.51 to 0.92; "
        "this pilot has only been evaluated in the calibration-record domain."
    )

    def __init__(self, reader_path: Path = SAE_READER, offsets_path: Path = SAE_OFFSETS) -> None:
        reader = np.load(reader_path, allow_pickle=False)
        offsets = np.load(offsets_path, allow_pickle=False)
        axes = [str(axis) for axis in reader["axes"].tolist()]
        layers = reader["reader_layers"].astype(np.int32)
        features = reader["feature_indices"].astype(np.int32)
        centers = reader["centers"].astype(np.float32)
        spans = reader["half_spans"].astype(np.float32)
        self.k = int(reader["k"][0])
        if not np.all(spans > 1e-6) or self.k <= 0:
            raise ValueError("SAE experimental reader has invalid calibration")
        offset_layers = offsets["layers"].astype(np.int32)
        offset_values = offsets["b_dec"].astype(np.float32)
        self.offsets = {
            int(layer): value for layer, value in zip(offset_layers, offset_values)
        }
        self.axes = axes
        self.by_axis = {
            axis: _SaeAxis(int(layer), int(feature), float(center), float(span))
            for axis, layer, feature, center, span in zip(
                axes, layers, features, centers, spans
            )
        }
        self.layers = sorted({item.layer for item in self.by_axis.values()})
        if any(layer not in self.offsets for layer in self.layers):
            raise ValueError("SAE runtime offsets do not cover all reader layers")
        self._weights: dict[int, dict[str, np.ndarray]] = {}

    def _layer_weights(self, layer: int) -> dict[str, np.ndarray]:
        if layer not in self._weights:
            source = np.load(SAE_DIR / f"saelens_topk_layer_{layer:02d}.npz", allow_pickle=False)
            loaded = {
                "W_enc": source["W_enc"].astype(np.float32),
                "b_enc": source["b_enc"].astype(np.float32),
                "W_dec_norm": source["W_dec_norm"].astype(np.float32),
            }
            if loaded["W_enc"].shape[1] != loaded["b_enc"].shape[0]:
                raise ValueError(f"SAE layer {layer} encoder dimensions disagree")
            self._weights[layer] = loaded
        return self._weights[layer]

    def observe(self, activations: dict[int, np.ndarray]) -> dict[str, Any]:
        # Exactly reproduce TopKTrainingSAE.encode: subtract b_dec, project,
        # multiply by decoder norms, select the K largest, then ReLU.
        feature_values: dict[tuple[int, int], float] = {}
        for layer in self.layers:
            weights = self._layer_weights(layer)
            hidden = (activations[layer] - self.offsets[layer]) @ weights["W_enc"]
            hidden = (hidden + weights["b_enc"]) * weights["W_dec_norm"]
            kth = float(np.partition(hidden, hidden.size - self.k)[-self.k])
            for item in self.by_axis.values():
                if item.layer != layer:
                    continue
                raw = float(hidden[item.feature])
                feature_values[(layer, item.feature)] = max(raw, 0.0) if raw >= kth else 0.0
        values: dict[str, dict[str, Any]] = {}
        for axis, item in self.by_axis.items():
            raw = feature_values[(item.layer, item.feature)]
            value = (raw - item.center) / item.half_span
            values[axis] = {
                "value": round(float(np.clip(value, -3.0, 3.0)), 4),
                "raw_value": round(float(value), 4),
                "layer": item.layer,
                "feature": item.feature,
            }
        return {
            "enabled": True,
            "label": self.label,
            "note": self.note,
            "values": values,
            "top_observed": _ranked(values)[:6],
            "scale": "0 is the sparse feature's calibration midpoint; ±1 is approximately half the endpoint separation.",
        }


class ExperimentalProbeSuite:
    """Load both optional research probes without affecting Direct-44 control."""

    def __init__(self) -> None:
        self.probes: dict[str, PersonaVectorProbe | SaeTopKProbe] = {}
        self.errors: dict[str, str] = {}
        for name, factory, paths in (
            ("persona", PersonaVectorProbe, (PERSONA_BANK,)),
            ("sae", SaeTopKProbe, (SAE_READER, SAE_OFFSETS)),
        ):
            if not all(path.exists() for path in paths):
                self.errors[name] = "Experimental bank not available"
                continue
            try:
                self.probes[name] = factory()
            except (OSError, ValueError, KeyError) as exc:
                self.errors[name] = str(exc)

    @property
    def layers(self) -> list[int]:
        return sorted({layer for probe in self.probes.values() for layer in probe.layers})

    def observe(self, activations: dict[int, np.ndarray]) -> dict[str, Any]:
        return {
            name: probe.observe(activations)
            for name, probe in self.probes.items()
        }

    def status(self) -> dict[str, Any]:
        return {
            "available": {name: True for name in self.probes},
            "labels": {name: probe.label for name, probe in self.probes.items()},
            "axis_counts": {name: len(probe.axes) for name, probe in self.probes.items()},
            "errors": self.errors,
            "layers": self.layers,
            "scope": "Direct-44 internal-experience prompt domain only",
            "persona_bank": str(PERSONA_BANK),
        }
