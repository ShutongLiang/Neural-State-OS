#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-12b-it-4bit-mlx"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "generated_runs"
DEFAULT_DATASET = (
    PROJECT_ROOT / "artifacts" / "dataset" / "direct6_composite.jsonl"
)
APPRAISAL_PROMPT = (
    "你正在接收一段第一人称内部体验记录。不要续写、评价或解释；"
    "只在内部重建这段体验所描述的身体与情感状态。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pool-last-tokens", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=("mlx", "cuda"),
        default="mlx",
        help="MLX is the original 12B path; CUDA uses the unquantized Transformers backend.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def fit_direction(
    activations: np.ndarray,
    records: list[dict],
    indices: list[int],
) -> np.ndarray:
    """Fit level covariance after removing scenario/style intercepts."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices:
        record = records[index]
        groups[(record["scenario_id"], record["style"])].append(index)
    covariance = np.zeros(activations.shape[-1], dtype=np.float64)
    for group_indices in groups.values():
        x = activations[group_indices].astype(np.float64)
        y = np.asarray(
            [records[index]["target_level"] for index in group_indices],
            dtype=np.float64,
        )
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean()
        covariance += np.sum(x * y[:, None], axis=0)
    return unit(covariance).astype(np.float32)


def evaluate_direction(
    direction: np.ndarray,
    activations: np.ndarray,
    records: list[dict],
    indices: list[int],
) -> tuple[float, float]:
    levels = np.asarray(
        [records[index]["target_level"] for index in indices],
        dtype=np.float64,
    )
    projections = (
        activations[indices].astype(np.float32) @ direction.astype(np.float32)
    )
    correlation = spearman(levels, projections)
    endpoints = {}
    for index, level in zip(indices, levels):
        endpoints[int(level)] = float(
            np.dot(activations[index].astype(np.float32), direction)
        )
    endpoint_correct = float(endpoints.get(6, 0.0) > endpoints.get(0, 0.0))
    return correlation, endpoint_correct


def capture_all(
    model,
    processor,
    records: list[dict],
    run_dir: Path,
    *,
    n_layers: int,
    capture_one,
    pool_last_tokens: int,
) -> tuple[np.memmap, dict]:
    activation_path = run_dir / "activations.float16.npy"
    progress_path = run_dir / "capture_progress.json"
    hidden_size = int(model.config.text_config.hidden_size)
    shape = (len(records), n_layers, hidden_size)
    if activation_path.exists():
        activations = np.lib.format.open_memmap(
            activation_path,
            mode="r+",
            dtype=np.float16,
            shape=shape,
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start = int(progress["completed"])
    else:
        activations = np.lib.format.open_memmap(
            activation_path,
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        start = 0
        progress = {
            "completed": 0,
            "shape": shape,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        progress_path.write_text(
            json.dumps(progress, indent=2),
            encoding="utf-8",
        )
    started = time.perf_counter()
    for index in range(start, len(records)):
        record = records[index]
        capture = capture_one(record["text"])
        activations[index] = capture.activations.astype(np.float16)
        if (index + 1) % 12 == 0 or index + 1 == len(records):
            activations.flush()
            progress.update(
                {
                    "completed": index + 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_this_session_seconds": (
                        time.perf_counter() - started
                    ),
                }
            )
            progress_path.write_text(
                json.dumps(progress, indent=2),
                encoding="utf-8",
            )
            print(
                f"[capture {index + 1:03d}/{len(records):03d}] "
                f"{record['axis']} {record['scenario_id']}",
                flush=True,
            )
    return activations, progress


def analyze(
    activations: np.ndarray,
    records: list[dict],
) -> tuple[dict, dict[str, np.ndarray]]:
    axes = sorted({record["axis"] for record in records})
    styles = sorted({record["style"] for record in records})
    n_layers = activations.shape[1]
    report = {"axes": {}, "styles": styles}
    full_directions = {}
    strict_layer_scores = {}
    for axis in axes:
        axis_indices = [
            index
            for index, record in enumerate(records)
            if record["axis"] == axis
        ]
        scenarios = sorted(
            {records[index]["scenario_id"] for index in axis_indices}
        )
        strict_scores = np.zeros(n_layers, dtype=np.float64)
        strict_endpoints = np.zeros(n_layers, dtype=np.float64)
        pooled_scores = np.zeros(n_layers, dtype=np.float64)
        fold_count = 0
        pooled_count = 0
        for heldout_scenario in scenarios:
            for heldout_style in styles:
                train_indices = [
                    index
                    for index in axis_indices
                    if records[index]["scenario_id"] != heldout_scenario
                    and records[index]["style"] != heldout_style
                ]
                test_indices = [
                    index
                    for index in axis_indices
                    if records[index]["scenario_id"] == heldout_scenario
                    and records[index]["style"] == heldout_style
                ]
                for layer in range(n_layers):
                    direction = fit_direction(
                        activations[:, layer, :],
                        records,
                        train_indices,
                    )
                    correlation, endpoint = evaluate_direction(
                        direction,
                        activations[:, layer, :],
                        records,
                        test_indices,
                    )
                    strict_scores[layer] += correlation
                    strict_endpoints[layer] += endpoint
                fold_count += 1
            train_indices = [
                index
                for index in axis_indices
                if records[index]["scenario_id"] != heldout_scenario
            ]
            for heldout_style in styles:
                test_indices = [
                    index
                    for index in axis_indices
                    if records[index]["scenario_id"] == heldout_scenario
                    and records[index]["style"] == heldout_style
                ]
                for layer in range(n_layers):
                    direction = fit_direction(
                        activations[:, layer, :],
                        records,
                        train_indices,
                    )
                    correlation, _ = evaluate_direction(
                        direction,
                        activations[:, layer, :],
                        records,
                        test_indices,
                    )
                    pooled_scores[layer] += correlation
                pooled_count += 1
        strict_scores /= fold_count
        strict_endpoints /= fold_count
        pooled_scores /= pooled_count
        best_layer = int(np.argmax(strict_scores))
        strict_layer_scores[axis] = strict_scores
        full_direction_layers = np.stack(
            [
                fit_direction(
                    activations[:, layer, :],
                    records,
                    axis_indices,
                )
                for layer in range(n_layers)
            ]
        )
        full_directions[axis] = full_direction_layers

        direction = full_direction_layers[best_layer]
        style_stds = []
        level_spans = []
        per_scenario_style = {}
        for scenario in scenarios:
            scenario_rows = [
                index
                for index in axis_indices
                if records[index]["scenario_id"] == scenario
            ]
            by_level = defaultdict(list)
            by_style = defaultdict(list)
            for index in scenario_rows:
                projection = float(
                    np.dot(
                        activations[index, best_layer].astype(np.float32),
                        direction,
                    )
                )
                by_level[records[index]["target_level"]].append(projection)
                by_style[records[index]["style"]].append(
                    (
                        records[index]["target_level"],
                        projection,
                    )
                )
            style_stds.extend(
                float(np.std(values)) for values in by_level.values()
            )
            level_means = [
                float(np.mean(by_level[level])) for level in range(7)
            ]
            level_spans.append(level_means[-1] - level_means[0])
            per_scenario_style[scenario] = {
                style: spearman(
                    np.asarray([item[0] for item in values]),
                    np.asarray([item[1] for item in values]),
                )
                for style, values in by_style.items()
            }
        report["axes"][axis] = {
            "scenario_count": len(scenarios),
            "record_count": len(axis_indices),
            "best_layer": best_layer,
            "strict_cross_scene_style_spearman": float(
                strict_scores[best_layer]
            ),
            "strict_endpoint_accuracy": float(
                strict_endpoints[best_layer]
            ),
            "pooled_style_cross_scene_spearman": float(
                pooled_scores[best_layer]
            ),
            "mean_same_state_style_std": float(np.mean(style_stds)),
            "mean_level_0_to_6_span": float(np.mean(level_spans)),
            "style_to_level_ratio": float(
                np.mean(style_stds)
                / max(abs(float(np.mean(level_spans))), 1e-12)
            ),
            "per_scenario_style_spearman_full_direction": (
                per_scenario_style
            ),
            "top_layers": [
                {
                    "layer": int(layer),
                    "strict_spearman": float(strict_scores[layer]),
                    "endpoint_accuracy": float(strict_endpoints[layer]),
                    "pooled_spearman": float(pooled_scores[layer]),
                }
                for layer in np.argsort(strict_scores)[::-1][:5]
            ],
        }
        print(
            f"[axis {axis}] layer={best_layer} "
            f"strict_rho={strict_scores[best_layer]:.3f} "
            f"endpoint={strict_endpoints[best_layer]:.3f}",
            flush=True,
        )

    common_scores = np.mean(
        np.stack([strict_layer_scores[axis] for axis in axes]),
        axis=0,
    )
    common_layer = int(np.argmax(common_scores))
    matrix = np.zeros((len(axes), len(axes)), dtype=np.float64)
    for left_index, left in enumerate(axes):
        for right_index, right in enumerate(axes):
            matrix[left_index, right_index] = float(
                np.dot(
                    full_directions[left][common_layer],
                    full_directions[right][common_layer],
                )
            )
    common_directions = np.stack(
        [full_directions[axis][common_layer] for axis in axes]
    ).astype(np.float64)
    gram = common_directions @ common_directions.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    orthogonal_directions = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-6)))
        @ eigenvectors.T
        @ common_directions
    )
    orthogonal_directions /= np.maximum(
        np.linalg.norm(orthogonal_directions, axis=1, keepdims=True),
        1e-12,
    )

    def cross_axis_leakage(directions: np.ndarray) -> np.ndarray:
        leakage = np.zeros((len(axes), len(axes)), dtype=np.float64)
        for target_index, target_axis in enumerate(axes):
            target_indices = [
                index
                for index, record in enumerate(records)
                if record["axis"] == target_axis
            ]
            groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            for index in target_indices:
                record = records[index]
                groups[(record["scenario_id"], record["style"])].append(
                    index
                )
            values = [[] for _ in axes]
            for group_indices in groups.values():
                levels = np.asarray(
                    [
                        records[index]["target_level"]
                        for index in group_indices
                    ],
                    dtype=np.float64,
                )
                hidden = activations[
                    group_indices,
                    common_layer,
                    :,
                ].astype(np.float32)
                for reader_index, direction in enumerate(directions):
                    values[reader_index].append(
                        spearman(levels, hidden @ direction.astype(np.float32))
                    )
            leakage[target_index] = [
                float(np.mean(reader_values))
                for reader_values in values
            ]
        return leakage

    raw_leakage = cross_axis_leakage(common_directions)
    orthogonal_leakage = cross_axis_leakage(orthogonal_directions)
    off_diagonal = ~np.eye(len(axes), dtype=bool)
    report["common_layer"] = {
        "layer": common_layer,
        "mean_strict_spearman": float(common_scores[common_layer]),
        "axis_order": axes,
        "direction_cosine_matrix": matrix.tolist(),
        "cross_axis_leakage": {
            "rows_are_varied_axis_columns_are_reader_axis": True,
            "raw_matrix": raw_leakage.tolist(),
            "raw_mean_diagonal": float(np.mean(np.diag(raw_leakage))),
            "raw_mean_abs_off_diagonal": float(
                np.mean(np.abs(raw_leakage[off_diagonal]))
            ),
            "orthogonalized_matrix": orthogonal_leakage.tolist(),
            "orthogonalized_mean_diagonal": float(
                np.mean(np.diag(orthogonal_leakage))
            ),
            "orthogonalized_mean_abs_off_diagonal": float(
                np.mean(np.abs(orthogonal_leakage[off_diagonal]))
            ),
        },
    }
    bank = {
        "axes": np.asarray(axes),
        "common_layer": np.asarray([common_layer], dtype=np.int32),
        "common_directions_raw": common_directions.astype(np.float32),
    }
    for axis in axes:
        bank[f"directions_{axis}"] = full_directions[axis].astype(
            np.float32
        )
        bank[f"strict_scores_{axis}"] = strict_layer_scores[axis].astype(
            np.float32
        )
    return report, bank


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.dataset)
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_id = datetime.now(timezone.utc).strftime(
            "factorial_axis_validation_%Y%m%dT%H%M%SZ"
        )
        run_dir = args.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "records.json"
    if not rows_path.exists():
        rows_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Loading model from {args.model} using {args.backend}", flush=True)
    started = time.perf_counter()
    if args.backend == "mlx":
        from mlx_vlm.utils import load

        from gemma4_capture import capture_appraisal_residuals

        model, processor = load(str(args.model))
        n_layers = len(model.layers)

        def capture_one(scene: str):
            return capture_appraisal_residuals(
                model,
                processor,
                scene=scene,
                system_text=APPRAISAL_PROMPT,
                pool_last_tokens=args.pool_last_tokens,
            )

    else:
        from gemma4_torch_backend import (
            capture_appraisal_residuals,
            install_residual_writers,
            load_gemma4_cuda,
        )

        model, processor = load_gemma4_cuda(str(args.model))
        writers = install_residual_writers(model)
        n_layers = len(writers)

        def capture_one(scene: str):
            return capture_appraisal_residuals(
                model,
                processor,
                writers,
                scene=scene,
                system_text=APPRAISAL_PROMPT,
                pool_last_tokens=args.pool_last_tokens,
            )

    print(
        f"Model loaded in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    activations, progress = capture_all(
        model,
        processor,
        records,
        run_dir,
        n_layers=n_layers,
        capture_one=capture_one,
        pool_last_tokens=args.pool_last_tokens,
    )
    report, bank = analyze(activations, records)
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "model": str(args.model),
            "record_count": len(records),
            "activation_shape": list(activations.shape),
            "pool_last_tokens": args.pool_last_tokens,
            "backend": args.backend,
            "capture_progress": progress,
            "method": (
                "Directions are level covariance vectors after removing "
                "scenario/style intercepts. Strict evaluation leaves out "
                "one scenario and one surface style simultaneously."
            ),
        }
    )
    report_path = run_dir / "factorial_axis_validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(run_dir / "factorial_axis_bank.npz", **bank)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "report": str(report_path),
                "common_layer": report["common_layer"],
                "axis_summary": {
                    axis: {
                        key: values[key]
                        for key in (
                            "best_layer",
                            "strict_cross_scene_style_spearman",
                            "strict_endpoint_accuracy",
                            "style_to_level_ratio",
                        )
                    }
                    for axis, values in report["axes"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
