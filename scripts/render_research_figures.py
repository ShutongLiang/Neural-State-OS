#!/usr/bin/env python3
"""Render publication-grade Neural State OS research figures from real artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from cmcrameri import cm as cmc
import scienceplots
import numpy as np


CATEGORY_ORDER = ["somatic", "basic_affect", "relational", "appraisal", "motivation"]
CATEGORY_LABEL = {
    "somatic": "Somatic",
    "basic_affect": "Basic affect",
    "relational": "Relational",
    "appraisal": "Appraisal",
    "motivation": "Motivation",
}
CATEGORY_COLOR = {
    "somatic": "#0C5DA5",
    "basic_affect": "#00B945",
    "relational": "#FF9500",
    "appraisal": "#FF2C00",
    "motivation": "#845B97",
}
BAND_COLOR = {"Early": "#0C5DA5", "Middle": "#FF9500", "Late": "#00B945"}
CATEGORY_LINESTYLE = {
    "somatic": "-",
    "basic_affect": "--",
    "relational": "-.",
    "appraisal": ":",
    "motivation": (0, (5, 1)),
}
CATEGORY_MARKER = {
    "somatic": "o",
    "basic_affect": "s",
    "relational": "^",
    "appraisal": "D",
    "motivation": "P",
}
GRAPHITE = "#000000"
MUTED = "#555555"
GRID = "#DCE3E8"
PAPER = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-root",
        type=Path,
        required=True,
        help="Path to a full research-artifact tree (raw activations are not distributed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "research",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.style.use(["science", "no-latex", "nature"])
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def pretty_axis(name: str) -> str:
    return name.replace("_", " ")


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=GRAPHITE,
    )


def clean_axes(ax: mpl.axes.Axes, grid_axis: str | None = None) -> None:
    ax.grid(False)
    ax.minorticks_on()


def category_layout(axes: np.ndarray, categories: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    index = {str(axis): i for i, axis in enumerate(axes)}
    cat_by_axis = {str(axis): str(cat) for axis, cat in zip(axes, categories)}
    ordered_names: list[str] = []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for category in CATEGORY_ORDER:
        names = sorted([name for name in index if cat_by_axis[name] == category])
        ordered_names.extend(names)
        spans.append((category, cursor, cursor + len(names)))
        cursor += len(names)
    return np.asarray([index[name] for name in ordered_names], dtype=int), spans


def load_data(root: Path) -> dict[str, object]:
    run = root / "artifacts/generated_runs/direct44_31b_capture_20260802"
    with np.load(run / "factorial_axis_bank.npz", allow_pickle=False) as source:
        axes = source["axes"].astype(str)
        common_layer = int(source["common_layer"][0])
        strict = np.stack([source[f"strict_scores_{axis}"] for axis in axes]).astype(float)
    with np.load(run / "band_candidates/direct6_band_candidates.npz", allow_pickle=False) as source:
        band_axes = source["axes"].astype(str)
        band_categories = source["categories"].astype(str)
        candidate_names = source["candidate_names"].astype(str)
        candidate_layers = source["candidate_layers"].astype(int)
        heldout_scores = source["heldout_scores"].astype(float)
    if not np.array_equal(axes, band_axes):
        raise ValueError("Factorial bank and band bank axis orders differ")
    with np.load(
        root / "artifacts/direct44_31b_reader/direct44_postwriter_live_affect_reader.npz",
        allow_pickle=False,
    ) as source:
        reader_axes = source["axes"].astype(str)
        reader_categories = source["categories"].astype(str)
        reader_scores = source["strict_spearman"].astype(float)
        reader_layers = source["reader_layers"].astype(int)
    with np.load(
        root / "artifacts/direct44_31b_reader/direct44_postwriter_response_calibration.npz",
        allow_pickle=False,
    ) as source:
        response_axes = source["axes"].astype(str)
        response = source["response_per_dose"].astype(float)
        response_by_context = source["response_by_context_per_dose"].astype(float)
    with np.load(
        root / "artifacts/direct44_31b_reader/direct44_postwriter_attribution_decoder.npz",
        allow_pickle=False,
    ) as source:
        transform = source["transform"].astype(float)
        ridge = float(source["ridge"][0])
    attr_report = json.loads(
        (root / "artifacts/direct44_31b_reader/direct44_postwriter_attribution_decoder.json").read_text()
    )
    validation_report = json.loads((run / "factorial_axis_validation_report.json").read_text())
    endpoint = np.asarray(
        [validation_report["axes"][axis]["strict_endpoint_accuracy"] for axis in axes], dtype=float
    )
    maps = {
        "reader": {axis: i for i, axis in enumerate(reader_axes)},
        "response": {axis: i for i, axis in enumerate(response_axes)},
    }
    reader_order = np.asarray([maps["reader"][axis] for axis in axes], dtype=int)
    response_order = np.asarray([maps["response"][axis] for axis in axes], dtype=int)
    reader_scores = reader_scores[reader_order]
    reader_layers = reader_layers[reader_order]
    if not np.array_equal(band_categories, reader_categories[reader_order]):
        raise ValueError("Reader and writer category labels differ")
    response = response[np.ix_(response_order, response_order)]
    response_by_context = response_by_context[:, response_order][:, :, response_order]
    transform = transform[np.ix_(response_order, response_order)]
    return {
        "axes": axes,
        "categories": band_categories,
        "common_layer": common_layer,
        "strict": strict,
        "candidate_names": candidate_names,
        "candidate_layers": candidate_layers,
        "heldout_scores": heldout_scores,
        "reader_scores": reader_scores,
        "reader_layers": reader_layers,
        "response": response,
        "response_by_context": response_by_context,
        "transform": transform,
        "ridge": ridge,
        "attribution": attr_report,
        "endpoint": endpoint,
    }


def add_pipeline_box(
    ax: mpl.axes.Axes,
    x: float,
    width: float,
    color: str,
    title: str,
    body: str,
) -> None:
    patch = FancyBboxPatch(
        (x, 0.18),
        width,
        0.64,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        facecolor=color,
        edgecolor="#87939D",
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, 0.67, title, transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.text(
        x + 0.018,
        0.50,
        body,
        transform=ax.transAxes,
        fontsize=7.5,
        color="#46525C",
        va="center",
        linespacing=1.45,
    )


def render_calibration_figure(data: dict[str, object], output_dir: Path) -> dict[str, float]:
    axes = data["axes"]
    categories = data["categories"]
    strict = data["strict"]
    candidate_layers = data["candidate_layers"]
    heldout_scores = data["heldout_scores"]
    fig = plt.figure(figsize=(7.2, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(
        1, 2,
        width_ratios=[2.25, 1.0],
        left=0.075, right=0.965, top=0.78, bottom=0.15,
        wspace=0.22,
    )
    fig.suptitle(
        "State directions emerge at distinct transformer depths",
        x=0.075,
        ha="left",
        fontsize=9,
        fontweight="normal",
        color=MUTED,
    )
    fig.text(
        0.075,
        0.855,
        "Factorial calibration across all 60 residual-stream layers of Gemma 4 31B",
        fontsize=7,
        color=MUTED,
    )

    ax_b = fig.add_subplot(grid[0, 0])
    panel_label(ax_b, "A", x=-0.08, y=1.02)
    layer_index = np.arange(strict.shape[1])
    for category in CATEGORY_ORDER:
        values = strict[categories == category]
        mean = values.mean(axis=0)
        ax_b.plot(
            layer_index, mean,
            color=CATEGORY_COLOR[category], linewidth=1.0,
            linestyle=CATEGORY_LINESTYLE[category], marker=CATEGORY_MARKER[category],
            markevery=5, markersize=2.8, markerfacecolor=CATEGORY_COLOR[category],
            markeredgewidth=0.5, label=CATEGORY_LABEL[category],
        )
    for boundary in (19.5, 39.5):
        ax_b.axvline(boundary, color="#75818B", linewidth=1.1, linestyle=(0, (4, 3)))
    ax_b.axvline(int(data["common_layer"]), color="#474747", linewidth=0.8, linestyle=(0, (3, 2)))
    ax_b.text(int(data["common_layer"]) + 0.8, 0.16, f"shared layer {int(data['common_layer'])}",
              color="#474747", fontsize=6.5, rotation=90, va="bottom")
    for center, label in [(9.5, "EARLY"), (29.5, "MIDDLE"), (49.5, "LATE")]:
        ax_b.text(center, 0.985, label, transform=ax_b.get_xaxis_transform(), ha="center",
                  va="top", fontsize=6.5, fontweight="bold", color=MUTED)
    ax_b.set_xlim(0, 59)
    ax_b.set_ylim(-0.10, 1.01)
    ax_b.set_xticks(np.arange(0, 60, 5))
    ax_b.set_xlabel("Transformer layer")
    ax_b.set_ylabel("Strict held-out Spearman ρ")
    ax_b.set_title("Where each state family becomes linearly recoverable", loc="left", pad=4)
    ax_b.legend(loc="lower right", ncol=2)
    clean_axes(ax_b, "y")

    ax_c = fig.add_subplot(grid[0, 1])
    panel_label(ax_c, "B", x=-0.15, y=1.02)
    rng = np.random.default_rng(7)
    positions = [2, 1, 0]
    for index, (position, name) in enumerate(zip(positions, ["Early", "Middle", "Late"])):
        values = candidate_layers[:, index]
        y = position + rng.normal(0, 0.075, size=len(values))
        ax_c.scatter(values, y, s=12, color=BAND_COLOR[name], alpha=1.0,
                     edgecolors="#000000", linewidths=0.25)
        ax_c.plot(
            [np.median(values)],
            [position],
            marker="D",
            markersize=4.5,
            color=BAND_COLOR[name],
            markeredgecolor=GRAPHITE,
            markeredgewidth=0.45,
            zorder=5,
        )
    ax_c.set_yticks(positions)
    ax_c.set_yticklabels(["Early bank", "Middle bank", "Late bank"])
    ax_c.set_xlim(-1, 60)
    ax_c.set_xlabel("Selected layer")
    ax_c.set_title("Axis-specific depth selection", loc="left", pad=4)
    clean_axes(ax_c, "x")
    ax_c.text(0.02, 0.04,
              "Mean held-out ρ\n"
              f"early  {heldout_scores[:, 0].mean():.3f}\n"
              f"middle {heldout_scores[:, 1].mean():.3f}\n"
              f"late   {heldout_scores[:, 2].mean():.3f}",
              transform=ax_c.transAxes, fontsize=6.3, linespacing=1.35,
              bbox={"boxstyle": "round,pad=0.55", "facecolor": PAPER,
                    "edgecolor": "#C7D0D7", "alpha": 0.96})
    outputs = save_figure(fig, output_dir / "figure3-calibration-depth")
    plt.close(fig)
    return {
        "strict_best_mean": float(strict.max(axis=1).mean()),
        "strict_best_min": float(strict.max(axis=1).min()),
        "band_early_mean": float(heldout_scores[:, 0].mean()),
        "band_middle_mean": float(heldout_scores[:, 1].mean()),
        "band_late_mean": float(heldout_scores[:, 2].mean()),
        "outputs": outputs,
    }


def render_validation_figure(data: dict[str, object], output_dir: Path) -> dict[str, object]:
    axes = data["axes"]
    categories = data["categories"]
    strict = data["strict"]
    reader_scores = data["reader_scores"]
    response = data["response"]
    endpoint = data["endpoint"]
    common_layer = int(data["common_layer"])
    attribution = data["attribution"]
    order, spans = category_layout(axes, categories)
    best = strict.max(axis=1)
    response_o = response[np.ix_(order, order)]

    fig = plt.figure(figsize=(7.2, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.05, 1.05, 1.35],
        left=0.07, right=0.965, top=0.82, bottom=0.16, wspace=0.34,
    )
    fig.suptitle(
        "State directions generalize beyond their calibration prompts",
        x=0.07,
        ha="left",
        fontsize=9,
        fontweight="bold",
        color=GRAPHITE,
    )
    fig.text(
        0.07,
        0.875,
        "Strict scene/style holdout · post-intervention readout · causal source attribution",
        fontsize=7,
        color=MUTED,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    panel_label(ax_a, "A", x=-0.18, y=1.02)
    rng = np.random.default_rng(19)
    mean_best = float(best.mean())
    minimum_best = float(best.min())
    strict_values = [best[categories == category] for category in CATEGORY_ORDER]
    box = ax_a.boxplot(strict_values, positions=np.arange(5), widths=0.56, patch_artist=True,
                       showfliers=False, medianprops={"color": GRAPHITE, "linewidth": 1.5},
                       whiskerprops={"color": "#7D8992"}, capprops={"color": "#7D8992"})
    for patch, category in zip(box["boxes"], CATEGORY_ORDER):
        patch.set_facecolor(PAPER); patch.set_alpha(1.0); patch.set_edgecolor(CATEGORY_COLOR[category]); patch.set_linewidth(0.8)
    for position, category in enumerate(CATEGORY_ORDER):
        values = best[categories == category]
        ax_a.scatter(position + rng.normal(0, 0.075, len(values)), values, s=14,
                     color=CATEGORY_COLOR[category], alpha=1.0, edgecolors="#000000", linewidths=0.25)
    ax_a.axhline(mean_best, color="#000000", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_a.set_xticks(np.arange(5))
    ax_a.set_xticklabels(["Somatic", "Affect", "Relational", "Appraisal", "Motivation"], rotation=28, ha="right")
    ax_a.set_ylim(0.82, 1.005)
    ax_a.set_ylabel("Strict held-out Spearman ρ")
    ax_a.set_title("Direction validity", loc="left", pad=4)
    ax_a.text(0.04, 0.06, f"mean  {mean_best:.3f}\nminimum  {minimum_best:.3f}\nendpoint accuracy  {endpoint.mean():.3f}",
              transform=ax_a.transAxes, fontsize=6.3, linespacing=1.35,
              bbox={"boxstyle": "round,pad=0.5", "facecolor": PAPER, "edgecolor": "#C7D0D7", "alpha": 0.96})
    clean_axes(ax_a, "y")

    ax_b = fig.add_subplot(grid[0, 1])
    panel_label(ax_b, "B", x=-0.18, y=1.02)
    category_values = [reader_scores[categories == category] for category in CATEGORY_ORDER]
    box = ax_b.boxplot(category_values, positions=np.arange(5), widths=0.56, patch_artist=True,
                       showfliers=False, medianprops={"color": GRAPHITE, "linewidth": 1.5},
                       whiskerprops={"color": "#7D8992"}, capprops={"color": "#7D8992"})
    for patch, category in zip(box["boxes"], CATEGORY_ORDER):
        patch.set_facecolor(PAPER); patch.set_alpha(1.0); patch.set_edgecolor(CATEGORY_COLOR[category]); patch.set_linewidth(0.8)
    for position, category in enumerate(CATEGORY_ORDER):
        values = reader_scores[categories == category]
        ax_b.scatter(position + rng.normal(0, 0.075, len(values)), values, s=14,
                     color=CATEGORY_COLOR[category], alpha=1.0, edgecolors="#000000", linewidths=0.25)
    common_mean = float(strict[:, common_layer].mean())
    reader_mean = float(reader_scores.mean())
    ax_b.axhline(reader_mean, color="#000000", linewidth=0.8, linestyle=(0, (4, 2)))
    ax_b.set_xticks(np.arange(5))
    ax_b.set_xticklabels(["Somatic", "Affect", "Relational", "Appraisal", "Motivation"], rotation=28, ha="right")
    ax_b.set_ylim(0.45, 1.005)
    ax_b.set_ylabel("Strict reader Spearman ρ")
    ax_b.set_title("Post-intervention readout", loc="left", pad=4)
    ax_b.text(0.04, 0.06, f"mean  {reader_mean:.3f}\nrange  {reader_scores.min():.3f}–{reader_scores.max():.3f}",
              transform=ax_b.transAxes, fontsize=6.3, linespacing=1.35,
              bbox={"boxstyle": "round,pad=0.5", "facecolor": PAPER, "edgecolor": "#C7D0D7", "alpha": 0.96})
    clean_axes(ax_b, "y")

    ax_d = fig.add_subplot(grid[0, 2])
    panel_label(ax_d, "C", x=-0.13, y=1.02)
    vmax = float(np.max(np.abs(response_o)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax_d.imshow(response_o, cmap=cmc.vik, norm=norm, interpolation="nearest", aspect="equal")
    ax_d.set_title("Measured writer → reader response", loc="left", pad=4)
    ax_d.set_xlabel("Injected writer source")
    ax_d.set_ylabel("Post-writer reader axis")
    centers = [(start + end - 1) / 2 for _, start, end in spans]
    labels = [CATEGORY_LABEL[category] for category, _, _ in spans]
    ax_d.set_xticks(centers)
    ax_d.set_xticklabels(labels, rotation=32, ha="right", fontsize=6.5)
    ax_d.set_yticks(centers)
    ax_d.set_yticklabels(labels, fontsize=6.5)
    for _, _, end in spans[:-1]:
        ax_d.axhline(end - 0.5, color=PAPER, linewidth=1.2)
        ax_d.axvline(end - 0.5, color=PAPER, linewidth=1.2)
    ax_d.plot([-0.5, 43.5], [-0.5, 43.5], color="#20262C", linewidth=0.55, alpha=0.65)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.035, pad=0.025)
    colorbar.set_label("Reader Δ per writer dose", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    top1 = int(attribution["heldout_context_top1"])
    top3 = int(attribution["heldout_context_top3"])
    trials = int(attribution["heldout_context_trials"])
    ax_d.text(
        0.02,
        0.985,
        f"Ridge source attribution\nTop-1  {top1}/{trials}\nTop-3  {top3}/{trials}",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PAPER, "edgecolor": "#B7C1C8", "alpha": 0.96},
    )

    fig.text(
        0.07,
        0.035,
        "A–B: strict held-out state validity/readout. C: writer-caused deltas only, not a decoder of total natural affect.",
        fontsize=6.3,
        color=MUTED,
    )
    outputs = save_figure(fig, output_dir / "figure4-validation-attribution")
    plt.close(fig)
    return {
        "strict_best_mean": mean_best,
        "strict_best_min": minimum_best,
        "endpoint_accuracy_mean": float(endpoint.mean()),
        "common_layer": common_layer,
        "common_layer_mean": common_mean,
        "reader_mean": reader_mean,
        "reader_min": float(reader_scores.min()),
        "reader_max": float(reader_scores.max()),
        "attribution_top1": top1,
        "attribution_top3": top3,
        "attribution_trials": trials,
        "outputs": outputs,
    }


def save_figure(fig: mpl.figure.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for suffix, kwargs in [
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
    ]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def main() -> None:
    args = parse_args()
    setup_style()
    data = load_data(args.backup_root)
    calibration = render_calibration_figure(data, args.output_dir)
    validation = render_validation_figure(data, args.output_dir)
    summary = {
        "source_root": str(args.backup_root),
        "figure3": calibration,
        "figure4": validation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "figure_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
