#!/usr/bin/env python3
"""Package a read-only Direct-44 hidden-activation reader.

This uses the clean Direct-44 capture only.  It intentionally selects the
best *readout* layer for each axis, which need not be one of the three causal
writer layers used by the demo.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    PROJECT_ROOT
    / "artifacts"
    / "generated_runs"
    / "direct44_full_capture_20260802"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "reader"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def endpoint_calibration(
    projections: np.ndarray,
    records: list[dict],
    indices: list[int],
) -> tuple[float, float]:
    """Return a robust 0→6 half-span and its signed median gap."""
    groups: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in indices:
        record = records[index]
        groups[(str(record["scenario_id"]), str(record["style"]))][
            int(record["target_level"])
        ].append(float(projections[index]))
    gaps = []
    for levels in groups.values():
        if 0 in levels and 6 in levels:
            gaps.append(float(np.mean(levels[6]) - np.mean(levels[0])))
    if not gaps:
        raise ValueError("No complete level-0/level-6 calibration groups")
    signed_gap = float(np.median(gaps))
    return max(abs(signed_gap) / 2.0, 1e-4), signed_gap


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing reader directory: {args.output_dir}"
        )
    bank_path = args.capture_run / "factorial_axis_bank.npz"
    report_path = args.capture_run / "factorial_axis_validation_report.json"
    records_path = args.capture_run / "records.json"
    activation_path = args.capture_run / "activations.float16.npy"
    for path in (bank_path, report_path, records_path, activation_path):
        if not path.exists():
            raise FileNotFoundError(path)

    bank = np.load(bank_path, allow_pickle=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    activations = np.load(activation_path, mmap_mode="r")
    axes = [str(value) for value in bank["axes"].tolist()]
    categories = [
        str(next(record["category"] for record in records if record["axis"] == axis))
        for axis in axes
    ]

    layers: list[int] = []
    directions: list[np.ndarray] = []
    half_spans: list[float] = []
    scores: list[float] = []
    rows = []
    for axis, category in zip(axes, categories):
        axis_report = report["axes"][axis]
        layer = int(axis_report["best_layer"])
        direction = bank[f"directions_{axis}"][layer].astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        indices = [
            index for index, record in enumerate(records) if record["axis"] == axis
        ]
        projections = activations[:, layer, :].astype(np.float32) @ direction
        half_span, signed_gap = endpoint_calibration(projections, records, indices)
        if signed_gap < 0:
            direction = -direction
        score = float(axis_report["strict_cross_scene_style_spearman"])
        layers.append(layer)
        directions.append(direction)
        half_spans.append(half_span)
        scores.append(score)
        rows.append(
            {
                "axis": axis,
                "category": category,
                "reader_layer": layer,
                "strict_cross_scene_style_spearman": score,
                "endpoint_half_span": half_span,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "direct44_live_affect_reader.npz"
    np.savez_compressed(
        output,
        axes=np.asarray(axes),
        categories=np.asarray(categories),
        reader_layers=np.asarray(layers, dtype=np.int32),
        directions=np.stack(directions).astype(np.float32),
        endpoint_half_spans=np.asarray(half_spans, dtype=np.float32),
        strict_spearman=np.asarray(scores, dtype=np.float32),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank": str(output),
        "source_capture": str(args.capture_run),
        "axis_count": len(axes),
        "method": (
            "For each Direct-44 axis, use the strict cross-scene/style "
            "reader-optimal layer and factorial direction.  Normalize live "
            "projections by a robust level-0→6 endpoint half-span.  This is "
            "a read-only estimator, separate from the three-layer writer."
        ),
        "axes": rows,
    }
    (args.output_dir / "reader_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bank": str(output),
                "axis_count": len(axes),
                "reader_layers": sorted(set(layers)),
                "mean_strict_spearman": float(np.mean(scores)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
