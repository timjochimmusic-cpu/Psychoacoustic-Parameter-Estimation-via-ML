"""Fold an existing average long-reference trajectory into one second."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAMETERS = {
    "loudness_zwtv": (500, "Loudness (Zwicker, TV) [sone]"),
    "sharpness_din_tv": (500, "Sharpness (DIN, TV) [acum]"),
    "roughness_dw": (10, "Roughness (Daniel & Weber) [asper]"),
    "tnr_ecma_perseg": (2, "TNR (ECMA, per-segment) [dB]"),
    "sii_ansi": (1, "SII (ANSI) [0-1]"),
}


def fold_average_to_one_second(input_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_csv)
    folded: dict[str, np.ndarray] = {}

    for name, (frames_per_second, _) in PARAMETERS.items():
        values = frame[name].dropna().to_numpy(dtype=np.float64)
        positions = np.arange(len(values)) % frames_per_second
        folded[name] = pd.Series(values).groupby(positions).mean().to_numpy()

    max_frames = max(len(values) for values in folded.values())
    output_frame = pd.DataFrame({"time_index": np.arange(max_frames)})
    for name, values in folded.items():
        output_frame[name] = np.pad(
            values,
            (0, max_frames - len(values)),
            constant_values=np.nan,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "parameter_average_one_second_all.csv"
    output_frame.to_csv(csv_path, index=False)

    figure, axes = plt.subplots(len(PARAMETERS), 1, figsize=(10, 15))
    figure.suptitle(
        "Average 60-second reference trajectories folded to one second"
    )
    for axis, (name, (frames_per_second, label)) in zip(
        axes, PARAMETERS.items()
    ):
        values = folded[name]
        times = np.arange(len(values)) / frames_per_second
        if len(values) == 1:
            axis.axhline(values[0], color="steelblue", linewidth=1.5)
            axis.scatter([0.5], values, color="steelblue", marker="x")
        else:
            axis.plot(times, values, color="steelblue", linewidth=1.2)
        axis.set_title(label)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Position within average second [s]")
        axis.set_ylabel("Mean value")
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    png_path = output_dir / "parameter_average_one_second_all.png"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    sharpness = folded["sharpness_din_tv"]
    sharpness_rate = PARAMETERS["sharpness_din_tv"][0]
    first_50_end = round(0.05 * sharpness_rate)
    stable_start = round(0.2 * sharpness_rate)
    first_50_mean = float(sharpness[:first_50_end].mean())
    stable_mean = float(sharpness[stable_start:].mean())
    full_mean = float(sharpness.mean())
    peak_index = int(sharpness.argmax())
    peak = float(sharpness[peak_index])
    metrics = pd.DataFrame(
        [
            {"metric": "first_50_ms_mean", "value": first_50_mean},
            {"metric": "stable_200_1000_ms_mean", "value": stable_mean},
            {"metric": "full_second_mean", "value": full_mean},
            {"metric": "peak", "value": peak},
            {"metric": "peak_time_ms", "value": peak_index / sharpness_rate * 1000},
            {
                "metric": "first_50_ms_above_stable_percent",
                "value": (first_50_mean / stable_mean - 1) * 100,
            },
            {
                "metric": "peak_above_stable_percent",
                "value": (peak / stable_mean - 1) * 100,
            },
            {
                "metric": "full_mean_above_stable_percent",
                "value": (full_mean / stable_mean - 1) * 100,
            },
        ]
    )
    metrics_path = output_dir / "sharpness_transient_impact_all.csv"
    metrics.to_csv(metrics_path, index=False)

    times = np.arange(len(sharpness)) / sharpness_rate
    transient_figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, sharpness, color="steelblue", label="Folded 60 s average")
    axis.axvspan(0, 0.05, color="coral", alpha=0.2, label="First 50 ms")
    axis.axhline(
        stable_mean,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Mean from 200–1000 ms",
    )
    axis.scatter(
        [peak_index / sharpness_rate],
        [peak],
        color="crimson",
        zorder=3,
    )
    axis.annotate(
        f"Peak: {peak:.3f} acum\n{(peak / stable_mean - 1) * 100:.1f}% above stable mean",
        xy=(peak_index / sharpness_rate, peak),
        xytext=(0.13, 0.88),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        fontsize=10,
    )
    axis.text(
        0.98,
        0.95,
        f"First 50 ms: {(first_50_mean / stable_mean - 1) * 100:.2f}% above stable\n"
        f"Full-second mean shift: {(full_mean / stable_mean - 1) * 100:.2f}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    axis.set_title("Sharpness transient after folding 60 seconds into one second")
    axis.set_xlabel("Position within average second [s]")
    axis.set_ylabel("Mean sharpness [acum]")
    axis.set_xlim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    transient_figure.tight_layout()
    transient_path = output_dir / "sharpness_transient_impact_all.png"
    transient_figure.savefig(transient_path, dpi=200, bbox_inches="tight")
    plt.close(transient_figure)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {transient_path}")


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_dir = root / "data" / "visualizations" / "music_dataset_60s"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_dir / "parameter_average_per_time_segment_all.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    fold_average_to_one_second(arguments.input_csv, arguments.output_dir)
