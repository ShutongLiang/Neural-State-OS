#!/usr/bin/env python3
"""Package a Direct-44 reader placed strictly after each axis's writer.

The earlier live reader chose its globally best semantic layer.  That is ideal
for recognising an experience record, but some such layers occur before the
axis's last causal writer layer and therefore cannot observe that write.  This
reader keeps the same clean factorial directions and validation protocol, but
selects the highest-scoring layer strictly after the final writer layer.
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
    PROJECT_ROOT / "artifacts" / "generated_runs" / "direct44_full_capture_20260802"
)
DEFAULT_WRITER = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "writer" / "direct44_raw_three_layer_writer.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "reader" / "direct44_postwriter_live_affect_reader.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--writer", type=Path, default=DEFAULT_WRITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def endpoint_half_span(
    projections: np.ndarray, records: list[dict], indices: list[int]
) -> tuple[float, float]:
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in indices:
        record = records[index]
        grouped[(str(record["scenario_id"]), str(record["style"]))][
            int(record["target_level"])
        ].append(float(projections[index]))
    gaps = [
        float(np.mean(levels[6]) - np.mean(levels[0]))
        for levels in grouped.values()
        if 0 in levels and 6 in levels
    ]
    if not gaps:
        raise ValueError("No full 0/6 endpoint calibration groups")
    signed = float(np.median(gaps))
    return max(abs(signed) / 2.0, 1e-4), signed


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing reader: {args.output}")
    for path in (args.writer, args.capture_run / "factorial_axis_bank.npz", args.capture_run / "factorial_axis_validation_report.json", args.capture_run / "records.json", args.capture_run / "activations.float16.npy"):
        if not path.exists():
            raise FileNotFoundError(path)
    writer = np.load(args.writer, allow_pickle=False)
    factor = np.load(args.capture_run / "factorial_axis_bank.npz", allow_pickle=False)
    report = json.loads((args.capture_run / "factorial_axis_validation_report.json").read_text(encoding="utf-8"))
    records = json.loads((args.capture_run / "records.json").read_text(encoding="utf-8"))
    activations = np.load(args.capture_run / "activations.float16.npy", mmap_mode="r")
    axes = [str(value) for value in writer["axes"].tolist()]
    categories = [str(value) for value in writer["categories"].tolist()]

    layers: list[int] = []
    directions: list[np.ndarray] = []
    spans: list[float] = []
    scores: list[float] = []
    final_writer_layers: list[int] = []
    details = []
    for index, (axis, category) in enumerate(zip(axes, categories)):
        final_writer_layer = int(np.max(writer["selected_layers"][index]))
        strict_scores = factor[f"strict_scores_{axis}"].astype(np.float32)
        candidates = np.arange(final_writer_layer + 1, len(strict_scores))
        if not len(candidates):
            raise ValueError(f"{axis}: no layer exists after writer layer {final_writer_layer}")
        layer = int(candidates[np.argmax(strict_scores[candidates])])
        direction = factor[f"directions_{axis}"][layer].astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        indices = [row for row, record in enumerate(records) if str(record["axis"]) == axis]
        projection = activations[:, layer, :].astype(np.float32) @ direction
        span, signed_gap = endpoint_half_span(projection, records, indices)
        if signed_gap < 0:
            direction = -direction
        layers.append(layer)
        directions.append(direction)
        spans.append(span)
        scores.append(float(strict_scores[layer]))
        final_writer_layers.append(final_writer_layer)
        details.append(
            {
                "axis": axis,
                "category": category,
                "last_writer_layer": final_writer_layer,
                "reader_layer": layer,
                "strict_cross_scene_style_spearman": float(strict_scores[layer]),
                "endpoint_half_span": span,
                "legacy_best_layer": int(report["axes"][axis]["best_layer"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        axes=np.asarray(axes),
        categories=np.asarray(categories),
        reader_layers=np.asarray(layers, dtype=np.int32),
        directions=np.stack(directions).astype(np.float32),
        endpoint_half_spans=np.asarray(spans, dtype=np.float32),
        strict_spearman=np.asarray(scores, dtype=np.float32),
        last_writer_layers=np.asarray(final_writer_layers, dtype=np.int32),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reader": str(args.output),
        "writer": str(args.writer),
        "source_capture": str(args.capture_run),
        "method": (
            "For every axis, select the highest strict cross-scene/style "
            "factorial reader layer strictly later than its final Direct-44 "
            "writer layer.  This reader is valid for observing live writes; "
            "the old semantic reader remains available for comparison."
        ),
        "mean_strict_spearman": float(np.mean(scores)),
        "minimum_strict_spearman": float(np.min(scores)),
        "axes": details,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"reader": str(args.output), "mean_score": float(np.mean(scores)), "min_score": float(np.min(scores))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
