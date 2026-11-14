#!/usr/bin/env python3
"""Assemble bounded Direct-44 writer→reader calibration shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTS = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "reader" / "postwriter_calibration_parts"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "reader" / "direct44_postwriter_response_calibration.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration: {args.output}")
    paths = sorted(args.parts_dir.glob("part_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No calibration parts in {args.parts_dir}")
    first = np.load(paths[0], allow_pickle=False)
    axes = [str(value) for value in first["reader_axes"].tolist()]
    contexts = first["contexts"]
    dose = float(first["dose"][0])
    response_by_context = np.empty(
        (len(contexts), len(axes), len(axes)), dtype=np.float32
    )
    covered: set[int] = set()
    for path in paths:
        part = np.load(path, allow_pickle=False)
        if (
            [str(value) for value in part["reader_axes"].tolist()] != axes
            or not np.array_equal(part["contexts"], contexts)
            or abs(float(part["dose"][0]) - dose) > 1e-6
        ):
            raise ValueError(f"Incompatible calibration part: {path}")
        indices = part["target_indices"].astype(np.int32)
        if any(int(index) in covered for index in indices):
            raise ValueError(f"Overlapping writer indices in {path}")
        covered.update(int(index) for index in indices)
        response_by_context[:, :, indices] = part["response_by_context_per_dose"]
    missing = sorted(set(range(len(axes))).difference(covered))
    if missing:
        raise ValueError(f"Calibration is missing writer columns: {missing}")
    response = np.median(response_by_context, axis=0).astype(np.float64)
    diagonal = np.diag(response)
    if np.any(np.abs(diagonal) < 1e-5):
        bad = [axis for axis, value in zip(axes, diagonal) if abs(value) < 1e-5]
        raise RuntimeError(f"Writer response has near-zero self effects: {bad}")
    inverse = np.linalg.inv(response)
    recovery = inverse @ response
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        axes=np.asarray(axes),
        dose=np.asarray([dose], dtype=np.float32),
        contexts=contexts,
        response_per_dose=response.astype(np.float32),
        response_by_context_per_dose=response_by_context,
        inverse=inverse.astype(np.float32),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parts": [str(path) for path in paths],
        "output": str(args.output),
        "dose": dose,
        "condition_number": float(np.linalg.cond(response)),
        "inverse_recovery_max_abs_error": float(
            np.max(np.abs(recovery - np.eye(len(axes))))
        ),
        "self_effects_per_dose": {axis: float(value) for axis, value in zip(axes, diagonal)},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
