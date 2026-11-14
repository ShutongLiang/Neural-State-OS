#!/usr/bin/env python3
"""Fit a small daily-chat-calibrated state reader for 31B.

The original Direct-44 reader is trained on an internal-experience prompt.
This experiment keeps that reader as a feature bank but observes it through
the standard Neural State OS chat template. It adds ordinary daily-chat null
messages plus paired counterfactual events, then learns a regularized joint
decoder for jealousy, anger, and anxiety.  It is an evaluation artifact only:
the raw reader and writer are not modified or overwritten.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "chat_state_calibration_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "gemma4-31b" / "reader" / "chat_state_v1"
STATE_AXES = ("jealousy", "anger", "anxiety")
RIDGE = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--url", default="http://127.0.0.1:18777")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    return parser.parse_args()


def request_observation(url: str, message: str) -> dict:
    payload = json.dumps(
        {"message": message, "thinking": False, "persona": "standard"}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/observe",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Observe request failed ({error.code}): {detail}") from error


def load_records(path: Path) -> list[dict]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Dataset is empty")
    required = {"record_id", "split", "condition", "axis", "level", "message"}
    for record in records:
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Record {record.get('record_id')} misses {sorted(missing)}")
        if record["axis"] is not None and record["axis"] not in STATE_AXES:
            raise ValueError(f"Unexpected target axis: {record['axis']}")
        if record["split"] not in {"train", "eval"}:
            raise ValueError(f"Unexpected split: {record['split']}")
    return records


def targets(records: list[dict]) -> np.ndarray:
    result = np.zeros((len(records), len(STATE_AXES)), dtype=np.float64)
    axis_index = {axis: index for index, axis in enumerate(STATE_AXES)}
    for index, record in enumerate(records):
        if record["axis"] is not None:
            result[index, axis_index[record["axis"]]] = float(record["level"])
    return result


def paired_accuracy(records: list[dict], values: np.ndarray, axis_index: int) -> tuple[int, int]:
    pairs: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        if record["split"] != "eval" or record["axis"] != STATE_AXES[axis_index]:
            continue
        prefix, suffix = record["record_id"].rsplit("_", 1)
        key = f"{prefix.replace('_control', '').replace('_target', '')}_{suffix}"
        bucket = pairs.setdefault(key, {})
        bucket["target" if record["level"] else "control"] = index
    complete = [pair for pair in pairs.values() if {"control", "target"}.issubset(pair)]
    wins = sum(values[pair["target"], axis_index] > values[pair["control"], axis_index] for pair in complete)
    return int(wins), len(complete)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if args.ridge <= 0:
        raise ValueError("ridge must be positive")
    records = load_records(args.dataset)
    print(f"Observing {len(records)} fresh standard-chat prompts", flush=True)
    raw_axes: list[str] | None = None
    features: list[list[float]] = []
    for index, record in enumerate(records, 1):
        observation = request_observation(args.url, record["message"])
        readout = observation["emotion_readout"]
        if not readout.get("enabled"):
            raise RuntimeError("Remote state reader is not enabled")
        values = readout["values"]
        if raw_axes is None:
            raw_axes = list(values)
        if list(values) != raw_axes:
            raise RuntimeError("Remote reader axis order changed during capture")
        features.append([float(values[axis]["raw_value"]) for axis in raw_axes])
        if index % 8 == 0 or index == len(records):
            print(f"[observe {index}/{len(records)}] {record['record_id']}", flush=True)
    assert raw_axes is not None
    raw = np.asarray(features, dtype=np.float64)
    y = targets(records)
    train = np.asarray([record["split"] == "train" for record in records])
    evaluation = ~train
    null_train = np.asarray(
        [record["split"] == "train" and record["condition"] == "daily_null" for record in records]
    )
    if int(train.sum()) < len(STATE_AXES) * 4 or int(null_train.sum()) < 6:
        raise ValueError("Insufficient train or daily-null records")
    null_mean = raw[null_train].mean(axis=0)
    null_scale = raw[null_train].std(axis=0, ddof=1)
    # A floor prevents an almost constant raw direction from becoming an
    # arbitrary high-gain feature after standardization.
    null_scale = np.maximum(null_scale, 0.15)
    x = (raw - null_mean) / null_scale
    x_train = x[train]
    y_train = y[train]
    feature_mean = x_train.mean(axis=0)
    centered = x_train - feature_mean
    weights = np.linalg.solve(
        centered.T @ centered + args.ridge * np.eye(centered.shape[1]),
        centered.T @ y_train,
    )
    intercept = y_train.mean(axis=0) - feature_mean @ weights
    decoded = x @ weights + intercept
    raw_target = np.column_stack([raw[:, raw_axes.index(axis)] for axis in STATE_AXES])
    report_axes = {}
    for axis_index, axis in enumerate(STATE_AXES):
        raw_wins, pair_count = paired_accuracy(records, raw_target, axis_index)
        decoded_wins, _ = paired_accuracy(records, decoded, axis_index)
        positives = evaluation & (y[:, axis_index] > 0.5)
        negatives = evaluation & (y[:, axis_index] < 0.5)
        threshold_accuracy = float(
            np.mean((decoded[evaluation, axis_index] >= 0.5) == (y[evaluation, axis_index] >= 0.5))
        )
        report_axes[axis] = {
            "raw_eval_pair_wins": raw_wins,
            "eval_pair_count": pair_count,
            "decoded_eval_pair_wins": decoded_wins,
            "decoded_eval_threshold_accuracy": threshold_accuracy,
            "decoded_eval_positive_mean": float(decoded[positives, axis_index].mean()),
            "decoded_eval_negative_mean": float(decoded[negatives, axis_index].mean()),
        }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "chat_state_disentangled_reader.npz",
        state_axes=np.asarray(STATE_AXES),
        raw_axes=np.asarray(raw_axes),
        daily_null_mean=null_mean.astype(np.float32),
        daily_null_scale=null_scale.astype(np.float32),
        weights=weights.astype(np.float32),
        intercept=intercept.astype(np.float32),
        ridge=np.asarray([args.ridge], dtype=np.float32),
    )
    np.savez_compressed(
        args.output_dir / "chat_state_calibration_observations.npz",
        raw_features=raw.astype(np.float32),
        decoded=decoded.astype(np.float32),
        targets=y.astype(np.float32),
        raw_axes=np.asarray(raw_axes),
        state_axes=np.asarray(STATE_AXES),
        records=np.asarray([json.dumps(record, ensure_ascii=False) for record in records]),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "remote_url": args.url,
        "prompt_domain": "fresh standard Neural State OS chat; no history, no writer, no generation",
        "state_axes": list(STATE_AXES),
        "raw_feature_count": len(raw_axes),
        "record_count": len(records),
        "train_count": int(train.sum()),
        "eval_count": int(evaluation.sum()),
        "daily_null_train_count": int(null_train.sum()),
        "method": (
            "Daily-chat-centered, jointly regularized linear state decoder over "
            "the 44 raw Direct-44 reader scores. It is a pilot monitor for three "
            "axes, not a replacement for raw hidden-state observation."
        ),
        "ridge": args.ridge,
        "axes": report_axes,
    }
    (args.output_dir / "chat_state_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
