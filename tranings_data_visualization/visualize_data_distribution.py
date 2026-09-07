import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = os.path.join(
    _ROOT, "data", "standardized_audio_files", "training_set",
    "all_psychoacoustic_labels.csv",
)
OUTPUT_DIR = os.path.join(_ROOT, "data", "standardized_audio_files", "training_set", "visualization")
TRAIN_DIR = os.path.join(_ROOT, "data", "standardized_audio_files", "training_set", "sound_files", "train")
VAL_DIR = os.path.join(_ROOT, "data", "standardized_audio_files", "training_set", "sound_files", "val")

PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]

PARAM_LABELS = {
    "loudness_zwtv": "Loudness (Zwicker, TV) [sone]",
    "sharpness_din_tv": "Sharpness (DIN, TV) [acum]",
    "roughness_dw": "Roughness (Daniel & Weber) [asper]",
    "tnr_ecma_perseg": "TNR (ECMA, per-segment) [dB]",
    "sii_ansi": "SII (ANSI) [0-1]",
}

CHUNK_SIZE = 50000


def _get_split_sources(train_dir: str, val_dir: str) -> tuple[set[str], set[str]]:
    """Returns the expected source_file names (i.e. '<stem>.csv') for the
    train and val WAV folders, based on calculate_reference_values.py's
    naming convention (<stem>.wav -> <stem>.csv)."""
    train_sources = {f"{p.stem}.csv" for p in Path(train_dir).glob("*.wav")}
    val_sources = {f"{p.stem}.csv" for p in Path(val_dir).glob("*.wav")}
    return train_sources, val_sources


def _split_df(df: pd.DataFrame, train_sources: set[str], val_sources: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["source_file"].isin(train_sources)]
    df_val = df[df["source_file"].isin(val_sources)]
    return df_train, df_val


def _load_chunked(filepath: str) -> pd.DataFrame:
    chunks = []
    for c in pd.read_csv(
        filepath,
        usecols=["time_index"] + PARAM_NAMES,
        dtype={"time_index": "int16"} | {p: "float32" for p in PARAM_NAMES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunks.append(c)
    return pd.concat(chunks, ignore_index=True)


def _load_with_source(filepath: str) -> pd.DataFrame:
    chunks = []
    for c in pd.read_csv(
        filepath,
        usecols=["source_file", "time_index"] + PARAM_NAMES,
        dtype={"source_file": "str", "time_index": "int16"} | {p: "float32" for p in PARAM_NAMES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunks.append(c)
    return pd.concat(chunks, ignore_index=True)


def plot_histograms(df: pd.DataFrame, output_dir: str, split_label: str):
    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    fig.suptitle(f"Parameter distributions — {split_label}")
    for ax, name in zip(axes, PARAM_NAMES):
        col = df[name].dropna().values
        ax.hist(col, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(PARAM_LABELS.get(name, name))
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")
    fig.tight_layout()
    path = os.path.join(output_dir, f"parameter_distributions_{split_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_average_per_time_segment(df: pd.DataFrame, output_dir: str, split_label: str):
    """Mean of each parameter at each time_index, for one split (train/val)."""
    grouped = df.groupby("time_index", observed=True)[PARAM_NAMES].mean()
    csv_path = os.path.join(output_dir, f"parameter_average_per_time_segment_{split_label}.csv")
    grouped.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    fig.suptitle(f"Average per time segment — {split_label}")
    for ax, name in zip(axes, PARAM_NAMES):
        pts = grouped[name].dropna()
        if len(pts) <= 2:
            ax.plot(pts.index.values, pts.values, color="steelblue",
                    marker="x", linestyle="", markersize=8)
        else:
            ax.plot(pts.index.values, pts.values, color="steelblue", linewidth=0.8)
        ax.set_title(PARAM_LABELS.get(name, name))
        ax.set_xlabel("Time index (frame)")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"parameter_average_per_time_segment_{split_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_length_distribution(df: pd.DataFrame, output_dir: str, split_label: str):
    """Per-file frame count for each parameter: min, max, avg + distribution."""
    csv_rows = []
    for name in PARAM_NAMES:
        col = df.groupby("source_file", observed=True)[name].count()
        csv_rows.append({"parameter": name, "min": col.min(), "max": col.max(), "mean": col.mean()})
    csv_path = os.path.join(output_dir, f"parameter_length_stats_{split_label}.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7),
                             gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"Frame length distribution — {split_label}")

    for i, name in enumerate(PARAM_NAMES):
        col = df.groupby("source_file", observed=True)[name].count()
        stats = {
            "Min": col.min(),
            "Max": col.max(),
            "Mean": col.mean(),
        }
        labels_bars = list(stats.keys())
        values_bars = list(stats.values())

        axes[0, i].bar(labels_bars, values_bars, color=["cornflowerblue", "coral", "seagreen"])
        axes[0, i].set_title(PARAM_LABELS.get(name, name))
        axes[0, i].set_ylabel("Frame count")

        axes[1, i].hist(col.values, bins=min(50, col.nunique()), color="steelblue",
                        edgecolor="white", linewidth=0.3, alpha=0.7)
        axes[1, i].set_xlabel("Frame count per file")
        axes[1, i].set_ylabel("Number of files")

        for label, v in zip(labels_bars, values_bars):
            axes[0, i].text(labels_bars.index(label), v + 0.5, f"{v:.1f}",
                            ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, f"parameter_length_stats_{split_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_value_stats(df: pd.DataFrame, output_dir: str, split_label: str):
    """Min, max, mean of actual parameter values for one split (train/val)."""
    stats = df[PARAM_NAMES].describe()
    csv_path = os.path.join(output_dir, f"parameter_value_stats_{split_label}.csv")
    stats.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    x = range(len(PARAM_NAMES))
    mins = stats.loc["min"].values
    maxs = stats.loc["max"].values
    means = stats.loc["mean"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Value stats — {split_label}")
    w = 0.25
    bars_min = ax.bar([i - w for i in x], mins, width=w, label="Min", color="cornflowerblue")
    bars_mean = ax.bar(x, means, width=w, label="Mean", color="seagreen")
    bars_max = ax.bar([i + w for i in x], maxs, width=w, label="Max", color="coral")

    for bars, vals in [(bars_min, mins), (bars_mean, means), (bars_max, maxs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2g}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS.get(n, n) for n in PARAM_NAMES], rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"parameter_value_stats_{split_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main(csv_path=CSV_PATH, output_dir=OUTPUT_DIR, all_data=False):
    os.makedirs(output_dir, exist_ok=True)

    if all_data:
        frame = _load_with_source(str(csv_path))
        if frame.empty:
            raise ValueError("No label rows available for distribution analysis")
        plot_histograms(frame, output_dir, "all")
        plot_average_per_time_segment(frame, output_dir, "all")
        plot_value_stats(frame, output_dir, "all")
        plot_length_distribution(frame, output_dir, "all")
        return

    train_sources, val_sources = _get_split_sources(TRAIN_DIR, VAL_DIR)
    print(f"train: {len(train_sources)} files — val: {len(val_sources)} files")

    print(f"Reading {csv_path} (with source_file, for splitting) ...")
    df2 = _load_with_source(csv_path)
    print(f"Loaded {len(df2):,} rows")

    df2_train, df2_val = _split_df(df2, train_sources, val_sources)
    df_train = df2_train.drop(columns=["source_file"])
    df_val = df2_val.drop(columns=["source_file"])
    del df2

    for split_label, split_df, split_df_with_source in [
        ("train", df_train, df2_train),
        ("val", df_val, df2_val),
    ]:
        print(f"\n=== {split_label}: {len(split_df):,} rows ===")
        plot_histograms(split_df, output_dir, split_label)
        plot_average_per_time_segment(split_df, output_dir, split_label)
        plot_value_stats(split_df, output_dir, split_label)
        plot_length_distribution(split_df_with_source, output_dir, split_label)

        print(f"\nSummary statistics ({split_label}):")
        print(split_df[PARAM_NAMES].describe().to_string())


if __name__ == "__main__":
    main()
