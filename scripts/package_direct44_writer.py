#!/usr/bin/env python3
"""Package the clean Direct-44 raw three-layer writer for the demo.

The release contains only raw factorial directions fitted from the clean
first-person Direct-44 corpus.  This is a transparent format conversion, not
an additional learned writer or a blend with an earlier experiment.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = PROJECT_ROOT / "artifacts" / "generated_runs" / "direct44_full_capture_20260802"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gemma4-12b" / "writer"
WEIGHTS = np.asarray((0.6, 0.6, 0.6), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    candidate_path = args.capture_run / "band_candidates" / "direct6_band_candidates.npz"
    source = np.load(candidate_path, allow_pickle=False)
    required = {
        "axes", "categories", "candidate_names", "candidate_layers", "directions", "heldout_scores",
    }
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"Candidate bank missing fields: {sorted(missing)}")
    selected_layers = source["candidate_layers"].astype(np.int32)
    directions = source["directions"].astype(np.float32)
    if selected_layers.shape[1] != 3 or directions.shape[1] != 3:
        raise ValueError("Expected exactly three early/middle/late candidates")
    if selected_layers.shape[:2] != directions.shape[:2]:
        raise ValueError("Layer/direction shape mismatch")
    if not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=-1) < 1e-6):
        raise ValueError("Directions must be finite, non-zero raw residual vectors")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "direct44_raw_three_layer_writer.npz"
    np.savez_compressed(
        output,
        axes=source["axes"],
        categories=source["categories"],
        selected_layers=selected_layers,
        directions=directions,
        layer_weights=np.broadcast_to(WEIGHTS, selected_layers.shape).copy(),
        heldout_scores=source["heldout_scores"].astype(np.float32),
        band_names=source["candidate_names"],
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank": str(output),
        "source_candidate_bank": str(candidate_path),
        "method": {
            "data": "clean Direct-44 first-person factorial corpus only",
            "directions": "raw factorial directions only",
            "selection": "highest strict cross-scene/style reader score in each early/middle/late band",
            "layer_weights": WEIGHTS.tolist(),
            "scope": "all_tokens",
            "included": [],
            "explicitly_excluded": [
                "legacy single-layer vectors",
                "Hybrid/Core/Natural blends",
                "subject/owner overlays",
                "anti-assistant overlays",
                "writer orthogonalization",
            ],
        },
        "axes": [
            {
                "axis": str(axis),
                "category": str(category),
                "layers": [int(value) for value in selected_layers[index]],
                "strict_scores": [float(value) for value in source["heldout_scores"][index]],
            }
            for index, (axis, category) in enumerate(zip(source["axes"].tolist(), source["categories"].tolist()))
        ],
    }
    (args.output_dir / "writer_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"bank": str(output), "axes": len(report["axes"]), "weights": WEIGHTS.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
