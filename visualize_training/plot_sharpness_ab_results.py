"""Create a presentation-ready summary of a sharpness A/B experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "isolated": "#D55E00",
    "contextual": "#0072B2",
    "mean_contextual_baseline": "#999999",
}
LABELS = {
    "isolated": "Isolated 1 s",
    "contextual": "Contextual 3 s",
    "mean_contextual_baseline": "Mean baseline",
}


def plot_results(results_dir: Path, output: Path) -> None:
    results_dir = Path(results_dir)
    overall = pd.read_csv(results_dir / "contextual_evaluation.csv")
    intervals = pd.read_csv(
        results_dir / "contextual_interval_evaluation.csv"
    )

    order = ["isolated", "mean_contextual_baseline", "contextual"]
    overall = overall.set_index("mode").loc[order].reset_index()
    interval_order = list(dict.fromkeys(intervals["interval_ms"]))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    figure.suptitle(
        "Sharpness prediction: isolated 1 s vs. 3 s context",
        fontsize=17,
        fontweight="bold",
    )

    bars = axes[0].bar(
        [LABELS[mode] for mode in overall["mode"]],
        overall["contextual_mse"],
        color=[COLORS[mode] for mode in overall["mode"]],
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    axes[0].set_title("Overall validation error")
    axes[0].set_ylabel("MSE against contextual reference (lower is better)")
    axes[0].set_ylim(0, overall["contextual_mse"].max() * 1.18)
    axes[0].grid(axis="y", alpha=0.25)
    improvement = 100 * (
        1
        - overall.set_index("mode").loc["contextual", "contextual_mse"]
        / overall.set_index("mode").loc["isolated", "contextual_mse"]
    )
    axes[0].text(
        0.98,
        0.95,
        f"{improvement:.1f}% lower MSE\nwith 3 s context",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["contextual"],
    )

    isolated = (
        intervals[intervals["mode"] == "isolated"]
        .set_index("interval_ms")
        .loc[interval_order]
    )
    contextual = (
        intervals[intervals["mode"] == "contextual"]
        .set_index("interval_ms")
        .loc[interval_order]
    )
    positions = list(range(len(interval_order)))
    width = 0.38
    axes[1].bar(
        [position - width / 2 for position in positions],
        isolated["contextual_mse"],
        width,
        label=LABELS["isolated"],
        color=COLORS["isolated"],
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        contextual["contextual_mse"],
        width,
        label=LABELS["contextual"],
        color=COLORS["contextual"],
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, [f"{value} ms" for value in interval_order])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Error by position within target second")
    axes[1].set_ylabel("MSE, logarithmic scale (lower is better)")
    axes[1].legend()
    axes[1].grid(axis="y", which="both", alpha=0.25)

    figure.text(
        0.5,
        0.01,
        "39 recordings | 312 examples | recording-grouped 80/20 split",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    plot_results(arguments.results_dir, arguments.output)
