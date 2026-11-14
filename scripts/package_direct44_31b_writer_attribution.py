#!/usr/bin/env python3
"""Package the 31B regularized writer-delta attribution decoder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "direct44_postwriter_response_calibration.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "direct44_postwriter_attribution_decoder.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge", type=float, default=3e-5)
    return parser.parse_args()


def ranks(transform: np.ndarray, response_by_context: np.ndarray) -> list[int]:
    result: list[int] = []
    for response in response_by_context:
        decoded = transform @ response
        for column in range(decoded.shape[1]):
            ranked = np.argsort(np.abs(decoded[:, column]))[::-1]
            result.append(int(np.where(ranked == column)[0][0]) + 1)
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing decoder: {args.output}")
    if args.ridge <= 0:
        raise ValueError("ridge must be positive")
    source = np.load(args.calibration, allow_pickle=False)
    required = {"axes", "response_per_dose", "response_by_context_per_dose", "dose"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"Calibration missing fields: {sorted(missing)}")
    response = source["response_per_dose"].astype(np.float64)
    response_by_context = source["response_by_context_per_dose"].astype(np.float64)
    if response.shape[0] != response.shape[1] or response_by_context.shape[1:] != response.shape:
        raise ValueError("Unexpected response matrix shape")
    transform = np.linalg.solve(
        response.T @ response + args.ridge * np.eye(response.shape[1]),
        response.T,
    )
    decoded_contexts = np.stack(
        [transform @ value for value in response_by_context], axis=0
    )
    matched = np.diagonal(decoded_contexts, axis1=1, axis2=2)
    rank_values = ranks(transform, response_by_context)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration": str(args.calibration),
        "output": str(args.output),
        "method": (
            "Tikhonov-regularized inverse of the measured 31B post-writer "
            "response matrix. It attributes a raw writer_delta to equivalent "
            "source axes only; it is not an affect-state reader."
        ),
        "ridge": args.ridge,
        "condition_number": float(np.linalg.cond(response)),
        "mean_matched_source_response": float(np.mean(matched)),
        "minimum_matched_source_response": float(np.min(matched)),
        "heldout_context_top1": int(sum(rank == 1 for rank in rank_values)),
        "heldout_context_top3": int(sum(rank <= 3 for rank in rank_values)),
        "heldout_context_trials": len(rank_values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        axes=source["axes"],
        dose=source["dose"].astype(np.float32),
        ridge=np.asarray([args.ridge], dtype=np.float64),
        transform=transform.astype(np.float32),
        response_per_dose=response.astype(np.float32),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
