"""Compare MoSQITo targets calculated with different amounts of context.

For each selected one-second target interval, this experiment compares:

1. MoSQITo run on the isolated one-second waveform.
2. The center second of a three-second MoSQITo calculation.
3. The same second sliced from an existing long-reference CSV.

Only sharpness and roughness are included because they are the current focus.
The script uses MoSQITo's returned timestamps for every slice and writes both
raw trajectories and timestamp-aligned comparison metrics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from mosqito import roughness_dw, sharpness_din_tv

from data_preprocessing.calculate_reference_values import (
    _extract_values_and_time,
)


PARAMETERS = {
    "sharpness_din_tv": sharpness_din_tv,
    "roughness_dw": roughness_dw,
}

DEFAULT_TARGET_STARTS_S = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def _calculate_trajectory(
    signal: np.ndarray,
    sample_rate: int,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray]:
    result = PARAMETERS[parameter](signal, sample_rate)
    values, times = _extract_values_and_time(
        param_name=parameter,
        result=result,
        signal_duration_s=len(signal) / sample_rate,
    )
    if times is None:
        raise ValueError(f"{parameter} did not return a time axis")
    return np.asarray(values, dtype=float), np.asarray(times, dtype=float)


def _slice_interval(
    values: np.ndarray,
    times: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return values in the half-open interval [start_s, end_s)."""
    mask = (times >= start_s) & (times < end_s)
    return values[mask], times[mask] - start_s


def _compute_target(
    audio_path: str,
    target_start_s: float,
) -> list[dict[str, object]]:
    signal, sample_rate = sf.read(audio_path)
    if signal.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {signal.shape}")

    target_start_sample = round(target_start_s * sample_rate)
    target_end_sample = round((target_start_s + 1.0) * sample_rate)
    context_start_sample = round((target_start_s - 1.0) * sample_rate)
    context_end_sample = round((target_start_s + 2.0) * sample_rate)

    isolated_signal = signal[target_start_sample:target_end_sample]
    context_signal = signal[context_start_sample:context_end_sample]

    expected_isolated = sample_rate
    expected_context = 3 * sample_rate
    if len(isolated_signal) != expected_isolated:
        raise ValueError(
            f"Target at {target_start_s:g}s has only {len(isolated_signal)} "
            f"samples; expected {expected_isolated}"
        )
    if len(context_signal) != expected_context:
        raise ValueError(
            f"Context at {target_start_s:g}s has only {len(context_signal)} "
            f"samples; expected {expected_context}"
        )

    rows: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        isolated_values, isolated_times = _calculate_trajectory(
            isolated_signal,
            sample_rate,
            parameter,
        )
        context_values, context_times = _calculate_trajectory(
            context_signal,
            sample_rate,
            parameter,
        )
        context_values, context_times = _slice_interval(
            context_values,
            context_times,
            1.0,
            2.0,
        )

        for method, values, times in (
            ("isolated_1s", isolated_values, isolated_times),
            ("center_of_3s", context_values, context_times),
        ):
            for frame_index, (time_s, value) in enumerate(zip(times, values)):
                rows.append(
                    {
                        "target_start_s": target_start_s,
                        "parameter": parameter,
                        "method": method,
                        "frame_index": frame_index,
                        "local_time_s": time_s,
                        "value": value,
                    }
                )
    return rows


def _load_long_reference_rows(
    reference_csv: Path,
    target_starts_s: list[float],
) -> list[dict[str, object]]:
    usecols = []
    for parameter in PARAMETERS:
        usecols.extend((f"{parameter}_time_s", parameter))
    frame = pd.read_csv(reference_csv, usecols=usecols)

    rows: list[dict[str, object]] = []
    for parameter in PARAMETERS:
        parameter_frame = frame[
            [f"{parameter}_time_s", parameter]
        ].dropna()
        all_times = parameter_frame[f"{parameter}_time_s"].to_numpy(float)
        all_values = parameter_frame[parameter].to_numpy(float)

        for target_start_s in target_starts_s:
            values, times = _slice_interval(
                all_values,
                all_times,
                target_start_s,
                target_start_s + 1.0,
            )
            for frame_index, (time_s, value) in enumerate(zip(times, values)):
                rows.append(
                    {
                        "target_start_s": target_start_s,
                        "parameter": parameter,
                        "method": "slice_of_60s",
                        "frame_index": frame_index,
                        "local_time_s": time_s,
                        "value": value,
                    }
                )
    return rows


def _aligned_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, float | int]:
    left = left.sort_values("local_time_s")
    right = right.sort_values("local_time_s")
    left_times = left["local_time_s"].to_numpy(float)
    left_values = left["value"].to_numpy(float)
    right_times = right["local_time_s"].to_numpy(float)
    right_values = right["value"].to_numpy(float)

    valid_left = np.isfinite(left_times) & np.isfinite(left_values)
    valid_right = np.isfinite(right_times) & np.isfinite(right_values)
    left_times, left_values = left_times[valid_left], left_values[valid_left]
    right_times, right_values = right_times[valid_right], right_values[valid_right]

    result: dict[str, float | int] = {
        "left_frames": len(left_times),
        "right_frames": len(right_times),
        "aligned_frames": 0,
        "mae": np.nan,
        "rmse": np.nan,
        "correlation": np.nan,
    }
    if len(left_times) == 0 or len(right_times) == 0:
        return result

    overlap = (left_times >= right_times.min()) & (
        left_times <= right_times.max()
    )
    comparison_times = left_times[overlap]
    comparison_left = left_values[overlap]
    if len(comparison_times) == 0:
        return result

    comparison_right = np.interp(
        comparison_times,
        right_times,
        right_values,
    )
    errors = comparison_left - comparison_right
    result["aligned_frames"] = len(errors)
    result["mae"] = float(np.mean(np.abs(errors)))
    result["rmse"] = float(np.sqrt(np.mean(errors**2)))

    if (
        len(errors) >= 2
        and np.std(comparison_left) > 0
        and np.std(comparison_right) > 0
    ):
        result["correlation"] = float(
            np.corrcoef(comparison_left, comparison_right)[0, 1]
        )
    return result


def _build_metrics(trajectories: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("isolated_1s", "center_of_3s"),
        ("isolated_1s", "slice_of_60s"),
        ("center_of_3s", "slice_of_60s"),
    )
    rows = []
    for (target_start_s, parameter), group in trajectories.groupby(
        ["target_start_s", "parameter"],
        sort=True,
    ):
        by_method = {
            method: method_group
            for method, method_group in group.groupby("method")
        }
        for left_method, right_method in comparisons:
            metrics = _aligned_metrics(
                by_method.get(left_method, pd.DataFrame()),
                by_method.get(right_method, pd.DataFrame()),
            )
            rows.append(
                {
                    "target_start_s": target_start_s,
                    "parameter": parameter,
                    "comparison": f"{left_method}_vs_{right_method}",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _build_transient_sensitivity(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Measure isolated-vs-context error after excluding startup time."""
    cutoffs_s = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    rows = []
    for parameter in PARAMETERS:
        parameter_frame = trajectories[trajectories["parameter"] == parameter]
        for cutoff_s in cutoffs_s:
            per_target = []
            for target_start_s, group in parameter_frame.groupby("target_start_s"):
                isolated = group[
                    (group["method"] == "isolated_1s")
                    & (group["local_time_s"] >= cutoff_s)
                ]
                context = group[group["method"] == "center_of_3s"]
                metrics = _aligned_metrics(isolated, context)
                per_target.append(metrics)
            rows.append(
                {
                    "parameter": parameter,
                    "excluded_start_s": cutoff_s,
                    "target_seconds": len(per_target),
                    "mean_mae": np.nanmean([row["mae"] for row in per_target]),
                    "mean_rmse": np.nanmean([row["rmse"] for row in per_target]),
                    "mean_correlation": np.nanmean(
                        [row["correlation"] for row in per_target]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_trajectories(
    trajectories: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {
        "isolated_1s": "tab:blue",
        "center_of_3s": "tab:orange",
        "slice_of_60s": "tab:green",
    }
    for target_start_s in sorted(trajectories["target_start_s"].unique()):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        for axis, parameter in zip(axes, PARAMETERS):
            selected = trajectories[
                (trajectories["target_start_s"] == target_start_s)
                & (trajectories["parameter"] == parameter)
            ]
            for method, group in selected.groupby("method", sort=False):
                group = group.sort_values("local_time_s")
                axis.plot(
                    group["local_time_s"],
                    group["value"],
                    label=f"{method} ({len(group)} frames)",
                    color=colors[method],
                    marker="o" if parameter == "roughness_dw" else None,
                    markersize=3,
                    linewidth=1.2,
                )
            axis.set_title(parameter)
            axis.set_xlabel("Time within target second [s]")
            axis.set_ylabel("MoSQITo value")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(
            f"Context comparison for source interval "
            f"{target_start_s:g}-{target_start_s + 1:g} s"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"context_comparison_{target_start_s:05.1f}s.png",
            dpi=160,
        )
        plt.close(fig)


def run_experiment(
    audio_path: Path,
    reference_csv: Path,
    output_dir: Path,
    target_starts_s: list[float],
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audio_path = Path(audio_path)
    reference_csv = Path(reference_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = sf.info(audio_path)
    duration_s = info.frames / info.samplerate
    invalid = [
        start
        for start in target_starts_s
        if start < 1.0 or start + 2.0 > duration_s
    ]
    if invalid:
        raise ValueError(
            "Every target needs one second of context on each side. "
            f"Invalid target starts for a {duration_s:g}s file: {invalid}"
        )

    calculated_rows: list[dict[str, object]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_compute_target, str(audio_path), start): start
                for start in target_starts_s
            }
            for future in as_completed(futures):
                start = futures[future]
                calculated_rows.extend(future.result())
                print(f"Calculated isolated/3s references for {start:g}-{start + 1:g}s")
    else:
        for start in target_starts_s:
            calculated_rows.extend(_compute_target(str(audio_path), start))
            print(f"Calculated isolated/3s references for {start:g}-{start + 1:g}s")

    long_rows = _load_long_reference_rows(reference_csv, target_starts_s)
    trajectories = pd.DataFrame(calculated_rows + long_rows).sort_values(
        ["target_start_s", "parameter", "method", "local_time_s"]
    )
    metrics = _build_metrics(trajectories)
    transient_sensitivity = _build_transient_sensitivity(trajectories)
    aggregate = (
        metrics.groupby(["parameter", "comparison"], as_index=False)
        .agg(
            target_seconds=("target_start_s", "count"),
            mean_left_frames=("left_frames", "mean"),
            mean_right_frames=("right_frames", "mean"),
            mean_aligned_frames=("aligned_frames", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_correlation=("correlation", "mean"),
        )
    )

    trajectories.to_csv(output_dir / "trajectories.csv", index=False)
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    transient_sensitivity.to_csv(
        output_dir / "transient_sensitivity.csv",
        index=False,
    )
    _plot_trajectories(trajectories, output_dir)

    print(f"Saved experiment outputs to {output_dir.resolve()}")
    return trajectories, metrics, aggregate


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_stem = (
        "[000001]_test_Bradley_Peck_-_living_like_no_tomorrow_"
        "00000-60000ms_ch1"
    )
    parser = argparse.ArgumentParser(
        description="Compare isolated, 3-second-context, and long-reference targets."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=root / "data" / "processed" / "1file" / f"{default_stem}.wav",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=root / "data" / "labels" / "test_1file" / f"{default_stem}.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "experiments" / "context_window_comparison",
    )
    parser.add_argument(
        "--target-starts",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_STARTS_S),
        metavar="SECONDS",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of target seconds calculated concurrently (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    _, _, aggregate_results = run_experiment(
        audio_path=arguments.audio,
        reference_csv=arguments.reference_csv,
        output_dir=arguments.output_dir,
        target_starts_s=arguments.target_starts,
        workers=arguments.workers,
    )
    print("\nAggregate comparison:")
    print(aggregate_results.to_string(index=False))
