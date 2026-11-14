#!/usr/bin/env python3
"""Export early/middle/late reader candidates from the clean Direct-6 bank."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CAPTURE = PROJECT_ROOT / "artifacts" / "capture"
DEFAULT_BANDS = (("early", 10, 19), ("middle", 20, 29), ("late", 30, 40))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bands",
        nargs=3,
        metavar="NAME:START:END",
        help=(
            "Override the three inclusive layer bands. For the 60-layer 31B "
            "run use: early:0:19 middle:20:39 late:40:59. Defaults preserve "
            "the historical 48-layer Direct-44 bands."
        ),
    )
    return parser.parse_args()


def parse_bands(values: list[str] | None, n_layers: int) -> tuple[tuple[str, int, int], ...]:
    if values is None:
        bands = DEFAULT_BANDS
    else:
        parsed = []
        for value in values:
            name, first, last = value.split(":", 2)
            parsed.append((name.strip(), int(first), int(last)))
        bands = tuple(parsed)
    if len(bands) != 3:
        raise ValueError("Exactly three selection bands are required")
    for name, first, last in bands:
        if not name or first < 0 or last < first or last >= n_layers:
            raise ValueError(f"Invalid layer band: {name}:{first}:{last} for {n_layers} layers")
    return bands


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    bank = np.load(args.capture_run / "factorial_axis_bank.npz", allow_pickle=False)
    records = json.loads((args.capture_run / "records.json").read_text(encoding="utf-8"))
    axes = [str(axis) for axis in bank["axes"].tolist()]
    sample_scores = bank[f"strict_scores_{axes[0]}"]
    bands = parse_bands(args.bands, len(sample_scores))
    categories = [
        next(str(row["category"]) for row in records if str(row["axis"]) == axis)
        for axis in axes
    ]
    selected_layers, selected_directions, selected_scores, axis_report = [], [], [], {}
    for axis in axes:
        scores = bank[f"strict_scores_{axis}"].astype(np.float32)
        layers, directions, values = [], [], []
        for label, first, last in bands:
            layer = first + int(np.argmax(scores[first : last + 1]))
            layers.append(layer)
            directions.append(bank[f"directions_{axis}"][layer].astype(np.float32))
            values.append(float(scores[layer]))
        selected_layers.append(layers)
        selected_directions.append(np.stack(directions))
        selected_scores.append(values)
        axis_report[axis] = {
            label: {"layer": int(layer), "strict_cross_scene_style_spearman": score}
            for (label, _first, _last), layer, score in zip(bands, layers, values)
        }
        print(axis + ": " + json.dumps(axis_report[axis], ensure_ascii=False), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "direct6_band_candidates.npz"
    np.savez_compressed(
        output,
        axes=np.asarray(axes),
        categories=np.asarray(categories),
        candidate_names=np.asarray([label for label, _first, _last in bands]),
        candidate_layers=np.asarray(selected_layers, dtype=np.int32),
        directions=np.asarray(selected_directions, dtype=np.float32),
        heldout_scores=np.asarray(selected_scores, dtype=np.float32),
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture_run": str(args.capture_run),
        "method": (
            "Clean Direct-6 factorial directions; strict score leaves out one "
            "scenario and one surface style. Candidates are not writers until "
            "they pass the separate +/- causal screen."
        ),
        "bands": [
            {"name": name, "minimum_layer": first, "maximum_layer": last}
            for name, first, last in bands
        ],
        "axes": axis_report,
        "bank": str(output),
    }
    (args.output_dir / "band_candidate_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SUMMARY=" + json.dumps({"bank": str(output), "axes": axis_report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
