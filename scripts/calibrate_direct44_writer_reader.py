#!/usr/bin/env python3
"""Calibrate writer→reader cross-talk on the actual 12B intervention path.

The Direct-44 corpus trains the reader, but residual writing can still carry
one axis into another reader at later layers.  This small calibration measures
that causal response directly: each writer is injected alone on neutral chat
prompts, then all 44 raw reader deltas are captured before generation.

The resulting inverse is for *writer attribution only*.  It must not replace
the raw affect readout, because raw co-activation is meaningful state evidence.
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

from server_mlx import Direct44Engine, MODEL_PATH, READER_PATH, SYSTEM_PROMPT, WRITER_PATH


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "gemma4-12b"
    / "reader"
    / "direct44_postwriter_response_calibration.npz"
)
CONTEXTS = [
    "事情按预期正常进行。我安静地注意到眼前的情况，准备自然回应。",
    "用户问我此刻的状态。我先平静地理解这句话，再准备回答。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--writer", type=Path, default=WRITER_PATH)
    parser.add_argument("--reader", type=Path, default=READER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dose", type=float, default=6.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Write a bounded writer-column shard instead of a full inverse.",
    )
    return parser.parse_args()


def normalised_delta(
    engine: Direct44Engine,
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
    if not args.reader.exists() or not args.writer.exists() or not args.model.exists():
        raise FileNotFoundError("Model, writer, and reader must all exist")
    if not 0.1 <= args.dose <= 20.0:
        raise ValueError("dose must be within [0.1, 20]")

    print(f"Loading model: {args.model}", flush=True)
    engine = Direct44Engine(
        args.model,
        args.writer,
        max_tokens=1,
        thinking_max_tokens=1,
        system_prompt=SYSTEM_PROMPT,
        reader_path=args.reader,
    )
    if not engine.reader_enabled:
        raise RuntimeError("The hidden activation reader did not load")
    axes = engine.reader_axes
    start = max(0, int(args.start_index))
    end = len(axes) if args.end_index is None else min(len(axes), int(args.end_index))
    if start >= end:
        raise ValueError(f"Invalid writer slice [{start}, {end})")
    target_axes = axes[start:end]
    response_by_context = np.zeros(
        (len(CONTEXTS), len(axes), len(target_axes)), dtype=np.float32
    )
    zero = {axis: 0.0 for axis in axes}
    for context_index, text in enumerate(CONTEXTS, 1):
        prompt = engine.prompt([{"role": "user", "content": text}], thinking=False)
        for layer in engine.circuit_layers.values():
            layer.clear_control()
        baseline = engine._capture_reader_projections(prompt)
        for local_index, axis in enumerate(target_axes, 1):
            engine.set_axes(
                {axis: args.dose},
                "all_tokens",
                {"mode": "raw", "headroom": 1.35},
            )
            written = engine._capture_reader_projections(prompt)
            response_by_context[context_index - 1, :, local_index - 1] = (
                normalised_delta(engine, written, baseline) / args.dose
            )
            if local_index % 4 == 0 or local_index == len(target_axes):
                print(
                    f"[context {context_index}/{len(CONTEXTS)}] "
                    f"writer={local_index}/{len(target_axes)} "
                    f"(global {start + local_index}/{len(axes)})",
                    flush=True,
                )
        engine.set_axes(zero, "all_tokens", {"mode": "raw", "headroom": 1.35})

    if args.partial:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            reader_axes=np.asarray(axes),
            target_axes=np.asarray(target_axes),
            target_indices=np.arange(start, end, dtype=np.int32),
            dose=np.asarray([args.dose], dtype=np.float32),
            contexts=np.asarray(CONTEXTS),
            response_by_context_per_dose=response_by_context,
        )
        print(
            json.dumps(
                {"partial": str(args.output), "targets": target_axes},
                ensure_ascii=False,
            )
        )
        return

    if start != 0 or end != len(axes):
        raise ValueError("A non-partial full calibration must cover all 44 writer axes")
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
            "chat prompts.  The matrix records all raw reader deltas per dose. "
            "Its inverse is used only to attribute a live writer_delta to the "
            "most likely source controls; it does not replace raw state readout."
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "condition_number": report["condition_number"],
                "inverse_recovery_max_abs_error": report["inverse_recovery_max_abs_error"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
