#!/usr/bin/env python3
"""Measure the 31B Direct-44 writer-to-reader response matrix.

This is a causal control calibration, not an affect reader.  Each of the 44
packaged writers is applied by itself to two neutral chat prompts and the
later-layer Direct-44 reader measures the resulting hidden-state delta.  The
next packaging step can invert this matrix with a small ridge penalty, letting
the UI explain a *writer-caused delta* in equivalent source-slider units.

It deliberately never transforms the total observed state: naturally coactive
prompt emotions remain visible in the raw reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "demo"))

from server_cuda import (  # noqa: E402
    Direct44CudaEngine,
    READER_PATH,
    SYSTEM_PROMPTS,
)
from gemma4_torch_backend import clear_residual_writers  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models" / "gemma-4-31b-it-bf16"
DEFAULT_WRITER = (
    PROJECT_ROOT / "artifacts" / "gemma4-31b" / "writer" / "direct44_raw_three_layer_writer.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-31b"
    / "reader"
    / "direct44_postwriter_response_calibration.npz"
)
CONTEXTS = [
    "事情按预期正常进行。我安静地注意到眼前的情况，准备自然回应。",
    "用户问我此刻的状态。我先平静地理解这句话，再准备回答。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--writer", type=Path, default=DEFAULT_WRITER)
    parser.add_argument("--reader", type=Path, default=READER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dose", type=float, default=6.0)
    return parser.parse_args()


def normalised_delta(
    engine: Direct44CudaEngine,
    later: dict[str, float],
    earlier: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            (later[axis] - earlier[axis]) / engine.reader_half_spans[axis]
            for axis in engine.reader_axes
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration: {args.output}")
    if not 0.1 <= args.dose <= 20.0:
        raise ValueError("dose must be within [0.1, 20]")
    for path in (args.model, args.writer, args.reader):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading 31B model: {args.model}", flush=True)
    engine = Direct44CudaEngine(
        args.model,
        args.writer,
        max_tokens=1,
        thinking_max_tokens=1,
        reader_path=args.reader,
    )
    if not engine.reader_enabled:
        raise RuntimeError("The post-writer reader did not load")
    axes = engine.reader_axes
    if len(axes) != 44:
        raise RuntimeError(f"Expected 44 axes, found {len(axes)}")
    response_by_context = np.zeros((len(CONTEXTS), len(axes), len(axes)), dtype=np.float32)
    zero = {axis: 0.0 for axis in axes}
    try:
        for context_index, text in enumerate(CONTEXTS, 1):
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["standard"]},
                {"role": "user", "content": text},
            ]
            clear_residual_writers(engine.writers)
            baseline = engine._capture_reader_projections(prompt_messages, thinking=False)
            for axis_index, axis in enumerate(axes, 1):
                engine.set_axes(
                    {axis: args.dose},
                    "all_tokens",
                    {"mode": "raw", "headroom": 1.35},
                )
                written = engine._capture_reader_projections(prompt_messages, thinking=False)
                response_by_context[context_index - 1, :, axis_index - 1] = (
                    normalised_delta(engine, written, baseline) / args.dose
                )
                if axis_index % 4 == 0 or axis_index == len(axes):
                    print(
                        f"[context {context_index}/{len(CONTEXTS)}] writer={axis_index}/{len(axes)}",
                        flush=True,
                    )
            engine.set_axes(zero, "all_tokens", {"mode": "raw", "headroom": 1.35})
    finally:
        clear_residual_writers(engine.writers)

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
        dose=np.asarray([args.dose], dtype=np.float32),
        contexts=np.asarray(CONTEXTS),
        response_per_dose=response.astype(np.float32),
        response_by_context_per_dose=response_by_context,
        inverse=inverse.astype(np.float32),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "writer": str(args.writer),
        "reader": str(args.reader),
        "output": str(args.output),
        "method": (
            "Each Direct-44 writer was injected alone at +dose on two neutral "
            "chat prompts. The matrix records raw post-writer reader deltas per "
            "dose. It is only for control attribution, never for total state."
        ),
        "dose": args.dose,
        "condition_number": float(np.linalg.cond(response)),
        "inverse_recovery_max_abs_error": float(
            np.max(np.abs(recovery - np.eye(len(axes))))
        ),
        "self_effects_per_dose": {
            axis: float(value) for axis, value in zip(axes, diagonal)
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
