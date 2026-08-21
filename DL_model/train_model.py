import random
import shutil
from datetime import datetime
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .DL_model import PsychoacousticModel
from .params import PARAM_NAMES
from .compute_loss import compute_loss
from visualize_training.visualize_training import hold_plot, plot_losses

def _load_time_biases(csv_path: str | Path) -> dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    counts = [500, 500, 9, 2, 1]
    biases: dict[str, torch.Tensor] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = df[name].dropna().values.astype(np.float32)
        b = torch.from_numpy(vals).float()
        T = counts[i]
        if b.numel() != T:
            b = F.interpolate(b.view(1, 1, -1), size=T, mode="linear", align_corners=False).view(-1)
        biases[name] = b
    return biases


def _get_device(device_id: int = 0) -> torch.device:
    """
    Get a suitable PyTorch device based on the availability of CUDA or DirectML. Fallback to CPU if neither works.

    Args:
        device_id (int): The index of the device to use. Defaults to 0.

    Returns:
        torch.device: The device to be used.
    """
    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
        idx = min(device_id, torch.cuda.device_count() - 1)
        print(f"Using CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
        return torch.device("cuda", idx)
    try:
        import torch_directml
        count = torch_directml.device_count()
        print(f"Available DirectML devices: {count}")
        for i in range(count):
            print(f"  {i}: {torch_directml.device_name(i)}")
        idx = min(device_id, count - 1)
        print(f"Using DirectML device {idx}: {torch_directml.device_name(idx)}")
        return torch_directml.device(idx)
    except ImportError:
        return torch.device("cpu")


class PsychoAcousticDataset(Dataset):
    """
    PyTorch dataset for paired audio files and psychoacoustic target values.

    The dataset loads psychoacoustic labels from a merged CSV file and groups
    them by source audio file. Parsed labels are cached as a PyTorch file to
    avoid repeatedly parsing large CSV files.

    Only labels corresponding to WAV files present in ``sound_dir`` are kept,
    allowing the same label file to be used for separate training and
    validation directories.

    Audio files are loaded into memory and optionally read in parallel using
    multiple worker threads. Loaded waveforms are cached in ``sound_dir`` to
    speed up subsequent dataset initialization.

    Each dataset item consists of a mono waveform tensor and a dictionary
    containing the corresponding time-dependent psychoacoustic target tensors.

    Parameters
    ----------
    sound_dir : Path
        Directory containing the WAV files belonging to the dataset split.
    csv_path : Path
        Path to the merged CSV file containing psychoacoustic labels.
    subset_indices : list[int] | None, optional
        Optional list of dataset indices to include.
    audio_workers : int, optional
        Number of threads used for loading audio files. A value of 0 or 1
        loads files sequentially.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        A waveform tensor and the corresponding psychoacoustic target tensors.
    """
    def __init__(self, sound_dir: Path, csv_path: Path, subset_indices: list[int] | None = None,
                 audio_workers: int = 0):
        self.sound_dir = Path(sound_dir)
        self.csv_path = Path(csv_path)

        labels_cache = self.csv_path.with_suffix(".labels.pt")
        if labels_cache.exists():
            print(f"Loading pre-parsed labels from {labels_cache}...")
            t0 = time.perf_counter()
            cached = torch.load(labels_cache, weights_only=True)
            all_stems = cached["stems"]
            all_labels = cached["labels"]
            print(f"  done in {time.perf_counter() - t0:.4f} s")
        else:
            print(f"Parsing {self.csv_path} into memory...")
            n = 0
            t0 = time.perf_counter()
            all_labels: dict[str, dict[str, torch.Tensor]] = {}
            usecols = ["source_file"] + PARAM_NAMES
            reader = pd.read_csv(self.csv_path, usecols=usecols, chunksize=100_000)
            for chunk in reader:
                chunk["stem"] = chunk["source_file"].str.replace(".csv", "", regex=False)
                for stem, grp in chunk.groupby("stem"):
                    if stem not in all_labels:
                        all_labels[stem] = {}
                    for name in PARAM_NAMES:
                        vals = grp[name].to_numpy(dtype=np.float32)
                        all_labels[stem].setdefault(name, []).append(torch.from_numpy(vals))
                n += len(chunk)
                print(f"  parsed {n / 1_000_000:.1f}M rows ({time.perf_counter() - t0:.2f}s)...")
            all_stems = sorted(all_labels.keys())
            for stem in all_stems:
                for name in PARAM_NAMES:
                    all_labels[stem][name] = torch.cat(all_labels[stem][name])
            elapsed = time.perf_counter() - t0
            print(f"  done in {elapsed:.4f} s — saving cache to {labels_cache}")
            torch.save({"stems": all_stems, "labels": all_labels}, labels_cache)

        # Filter to only the stems whose WAV file actually exists in sound_dir.
        # Necessary after split_train_val.py has moved files into train/ or
        # val/ subfolders — the cached/parsed label list still references
        # ALL original stems, regardless of which physical folder they now
        # live in.
        available_stems = {p.stem for p in self.sound_dir.glob("*.wav")}
        missing_before_filter = len(all_stems)
        all_stems = [s for s in all_stems if s in available_stems]
        n_filtered = missing_before_filter - len(all_stems)
        if n_filtered > 0:
            print(f"  Filtered out {n_filtered} stems not present in {self.sound_dir} "
                  f"(likely moved to a different split folder)")

        if subset_indices is not None:
            self.stems = [all_stems[i] for i in subset_indices]
            self._labels = {s: all_labels[s] for s in self.stems}
        else:
            self.stems = all_stems
            self._labels = {s: all_labels[s] for s in self.stems}

        audio_cache = self.sound_dir / "_audio_cache.pt"
        if audio_cache.exists():
            print(f"Loading pre-parsed audio from {audio_cache}...")
            t0 = time.perf_counter()
            self._waveforms = torch.load(audio_cache, weights_only=True)
            print(f"  done in {time.perf_counter() - t0:.4f} s")
        else:
            print(f"Loading {len(self.stems)} audio files into memory...")
            self._waveforms: dict[str, torch.Tensor] = {}
            t0 = time.perf_counter()
            total = len(self.stems)

            def _load_one(stem: str) -> tuple[str, torch.Tensor]:
                wav_path = self.sound_dir / f"{stem}.wav"
                audio, _ = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return stem, torch.from_numpy(audio).float().unsqueeze(0)

            if audio_workers > 1:
                import concurrent.futures
                n_workers = min(audio_workers, total)
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_load_one, stem): stem for stem in self.stems}
                    for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
                        stem, waveform = f.result()
                        self._waveforms[stem] = waveform
                        if i % 100 == 0 or i == total:
                            print(f"  {i}/{total} done ({time.perf_counter() - t0:.4f}s)")
            else:
                for i, stem in enumerate(self.stems):
                    stem, waveform = _load_one(stem)
                    self._waveforms[stem] = waveform
                    if (i + 1) % 100 == 0 or i + 1 == total:
                        print(f"  {i + 1}/{total} done ({time.perf_counter() - t0:.4f}s)")
            load_time = time.perf_counter() - t0
            print(f"  done in {load_time:.4f} s")
            if not subset_indices:
                torch.save(self._waveforms, audio_cache)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stem = self.stems[idx]
        return self._waveforms[stem], self._labels[stem]



def _collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    waveforms = [b[0] for b in batch]
    max_wav_len = max(w.shape[-1] for w in waveforms)
    batch_waveforms = []
    for w in waveforms:
        pad = max_wav_len - w.shape[-1]
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        batch_waveforms.append(w)
    waveform_batch = torch.stack(batch_waveforms, dim=0)

    max_frames = {
        name: max(b[1][name].shape[-1] for b in batch)
        for name in PARAM_NAMES
    }
    target_batch: dict[str, list[torch.Tensor]] = {n: [] for n in PARAM_NAMES}
    for b in batch:
        for name in PARAM_NAMES:
            t = b[1][name]
            pad = max_frames[name] - t.shape[-1]
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad), value=float("nan"))
            target_batch[name].append(t)
    targets = {
        name: torch.stack(target_batch[name], dim=0) for name in PARAM_NAMES
    }

    return waveform_batch, targets


def _training_step(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    targets: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_stats_path: Path,
) -> dict[str, float]:
    """Single forward-backward-update step for one batch."""
    waveform = waveform.to(device)
    targets = {name: t.to(device) for name, t in targets.items()}
    preds = model(waveform)
    trimmed = {name: targets[name][:, :preds[name].shape[-1]]
               for name in PARAM_NAMES}
    losses = compute_loss(model, preds, trimmed, stats_path=loss_stats_path)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    return {k: v.item() for k, v in losses.items()}


def _resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
) -> tuple[int, list[dict[str, float]]]:
    """Load the latest checkpoint if available, else start from scratch."""
    ckpt_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        return 0, []

    latest = ckpt_files[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {latest.name} (epoch {ckpt['epoch']})")
        return ckpt["epoch"], ckpt["history"]
    except Exception as e:
        print(f"Could not resume from {latest.name} ({e}) — starting from scratch")
        return 0, []


def _save_runtime_csv(output_dir: Path, stem: str, meta: dict, preds: dict[str, torch.Tensor], epoch_tag: str = "", targets: dict[str, torch.Tensor] | None = None):
    """Save per-parameter prediction statistics to CSV."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    rows = []
    for name in PARAM_NAMES:
        p = preds[name][0]
        if targets is not None:
            t = targets[name][:p.shape[-1]]
            mask = ~torch.isnan(t)
            if mask.any():
                p = p[mask]
        row = dict(meta)
        row["parameter"] = name
        row["output_frames"] = p.shape[-1]
        row["pred_min"] = round(p.min().item(), 6)
        row["pred_max"] = round(p.max().item(), 6)
        row["pred_mean"] = round(p.mean().item(), 6)
        rows.append(row)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}{ts}_{stem}_runtime.csv", index=False)


def _save_prediction_plots(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor], epoch_tag: str = ""):
    """Plot prediction vs target for each parameter and save as PNG."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]

        p_np, t_np = p.cpu().numpy(), t.numpy()
        n_frames = len(p_np)
        p_masked = np.ma.masked_where(np.isnan(t_np), p_np)

        fig, ax = plt.subplots(figsize=(10, 4))
        if n_frames == 1:
            ax.axhline(y=t_np[0], label="target", color="tab:blue", alpha=0.8)
            ax.axhline(y=p_np[0], label="prediction", color="tab:red", alpha=0.8)
        else:
            ax.plot(t_np, label="target", color="tab:blue", alpha=0.8)
            ax.plot(p_masked, label="prediction", color="tab:red", alpha=0.8)
        ax.set_xlabel("Time frame")
        ax.set_ylabel(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{stem}_{name}.png")
        plt.close(fig)


def _save_comparison_csv(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor], epoch_tag: str = ""):
    """Save side-by-side target vs prediction values for all params as CSV."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    all_arrays: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]
        p_np = p.cpu().numpy().copy()
        p_np[np.isnan(t.numpy())] = np.nan
        all_arrays[f"{name}_target"] = t.numpy()
        all_arrays[f"{name}_pred"] = p_np
    max_len = max(arr.shape[-1] for arr in all_arrays.values())
    df_dict: dict[str, np.ndarray] = {"time_index": np.arange(max_len)}
    for key, arr in all_arrays.items():
        pad = max_len - arr.shape[-1]
        df_dict[key] = np.pad(arr, (0, pad), constant_values=np.nan)
    pd.DataFrame(df_dict).to_csv(output_dir / f"{prefix}{stem}_comparison.csv", index=False)


def _load_epoch_predictions(
    dataset: PsychoAcousticDataset,
    checkpoint_dir: Path,
    device: torch.device,
    epoch: int | str,
) -> tuple[str, dict[str, dict[str, torch.Tensor]]] | None:
    """Load one checkpoint and return predictions grouped by stem."""
    stats_path = (
        Path(__file__).parent.parent
        / "data"
        / "standardized_audio_files"
        / "training_set"
        / "visualization"
        / "parameter_average_per_time_segment_train.csv"
    )

    biases = _load_time_biases(stats_path)
    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)

    if epoch == "newest":
        ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
        if not ckpt_files:
            print("No checkpoint found — skipping comparison")
            return None
        ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path.name} not found — skipping")
            return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch_label = ckpt_path.stem.replace("epoch_", "Epoch ")
    print(f"Loaded {ckpt_path.name} for comparison")

    predictions: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            waveform, _ = dataset[idx]
            inp = waveform.unsqueeze(0).to(device)
            preds = model(inp)
            stem = dataset.stems[idx]
            predictions[stem] = {
                name: preds[name][0].detach().cpu() for name in PARAM_NAMES
            }

    return epoch_label, predictions


def _save_multi_epoch_plots(
    output_dir: Path,
    dataset: PsychoAcousticDataset,
    epoch_predictions: dict[str, dict[str, dict[str, torch.Tensor]]],
):
    """Save individual plots and one subplot figure per stem."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_colors = plt.get_cmap("Set1").colors

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        stem = dataset.stems[idx]

        # Collect parameters that actually have predictions
        available_params = []

        for name in PARAM_NAMES:
            available = {
                label: preds_by_stem[stem][name]
                for label, preds_by_stem in epoch_predictions.items()
                if stem in preds_by_stem
            }

            if available:
                available_params.append((name, available))

        if not available_params:
            continue

        # --------------------------------------------------
        # Individual plots
        # --------------------------------------------------
        for name, available in available_params:
            max_pred_len = max(pred.shape[-1] for pred in available.values())
            reference = target[name][:max_pred_len].cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 4))

            if len(reference) == 1:
                ax.axhline(
                    reference[0],
                    label="Reference",
                    color="black",
                )

                for i, (label, pred) in enumerate(available.items()):
                    ax.axhline(
                        pred[0].item(),
                        label=label,
                        color=prediction_colors[i % len(prediction_colors)],
                    )
            else:
                ax.plot(
                    reference,
                    label="Reference",
                    color="black",
                )

                for i, (label, pred) in enumerate(available.items()):
                    pred_np = pred.numpy()
                    ref_for_pred = target[name][:len(pred_np)].cpu().numpy()

                    pred_masked = np.ma.masked_where(
                        np.isnan(ref_for_pred),
                        pred_np,
                    )

                    ax.plot(
                        pred_masked,
                        label=label,
                        color=prediction_colors[i % len(prediction_colors)],
                    )

            ax.set_xlabel("Time frame")
            ax.set_ylabel(name)
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                output_dir / f"{stem}_{name}_comparison.png",
                dpi=300,
            )
            plt.close(fig)

        # --------------------------------------------------
        # Combined subplot figure
        # --------------------------------------------------
        n_plots = len(available_params)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 4 * n_rows),
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for ax, (name, available) in zip(axes_flat, available_params):
            max_pred_len = max(
                pred.shape[-1]
                for pred in available.values()
            )

            reference = (
                target[name][:max_pred_len]
                .cpu()
                .numpy()
            )

            if len(reference) == 1:
                ax.axhline(
                    reference[0],
                    label="Reference",
                    color="black",
                )

                for i, (label, pred) in enumerate(available.items()):
                    ax.axhline(
                        pred[0].item(),
                        label=label,
                        color=prediction_colors[i % len(prediction_colors)],
                    )
            else:
                ax.plot(
                    reference,
                    label="Reference",
                    color="black",
                )

                for i, (label, pred) in enumerate(available.items()):
                    pred_np = pred.numpy()
                    ref_for_pred = (
                        target[name][:len(pred_np)]
                        .cpu()
                        .numpy()
                    )

                    pred_masked = np.ma.masked_where(
                        np.isnan(ref_for_pred),
                        pred_np,
                    )

                    ax.plot(
                        pred_masked,
                        label=label,
                        color=prediction_colors[i % len(prediction_colors)],
                    )

            ax.set_title(name)
            ax.set_xlabel("Time frame")
            ax.set_ylabel(name)
            ax.legend()

        # Hide unused subplot axes
        for ax in axes_flat[len(available_params):]:
            ax.set_visible(False)

        fig.suptitle(stem)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig.savefig(
            output_dir / f"{stem}_all_parameters_comparison.png",
            dpi=300,
        )

        plt.close(fig)

class _DatasetView(Dataset):
    """Lightweight view over a subset of a PsychoAcousticDataset, sharing the underlying data."""
    def __init__(self, dataset: PsychoAcousticDataset, indices: list[int]):
        n = len(dataset)
        for i in indices:
            if not 0 <= i < n:
                raise IndexError(
                    f"subset index {i} is out of range for dataset of size {n}"
                )
        self._dataset = dataset
        self._indices = indices
        self.stems = [dataset.stems[i] for i in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]


def run_comparison(
    sound_dir: Path,
    labels_csv_path: Path,
    checkpoint_dir: Path,
    n_samples: int = 1,
    device_id: int = 0,
    subset_indices: list[int] | None = None,
    epochs: list[int | str] | None = None,
    dataset: PsychoAcousticDataset | None = None,
    audio_workers: int = 0,
):
    """Compare epoch 0, selected epochs, and the reference in shared plots.

    Parameters
    ----------
    epochs : list[int | str] | None
        Epoch numbers to compare. Use "newest" for the latest checkpoint.
        Defaults to [0, "newest"].
    """
    if epochs is None:
        epochs = [0, "newest"]
    print("=" * 100)
    device = _get_device(device_id)
    if subset_indices is not None:
        if dataset is not None:
            dataset = _DatasetView(dataset, subset_indices)
        else:
            dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, subset_indices=subset_indices,
                                            audio_workers=audio_workers)
    elif dataset is None:
        dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, audio_workers=audio_workers)

    if n_samples > 0 and n_samples < len(dataset):
        dataset = _DatasetView(dataset, list(range(n_samples)))

    output_dir = Path(__file__).resolve().parent / "comparison"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Epoch 0 is always the baseline. Additional entries are compared against it.
    requested_epochs: list[int | str] = [0]
    requested_epochs.extend(ep for ep in epochs if ep != 0)

    epoch_predictions: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for ep in requested_epochs:
        result = _load_epoch_predictions(dataset, checkpoint_dir, device, ep)
        if result is not None:
            epoch_label, predictions = result
            epoch_predictions[epoch_label] = predictions

    _save_multi_epoch_plots(output_dir, dataset, epoch_predictions)
    print(f"Comparison plots saved to {output_dir}")
    hold_plot()


def _log_epoch(
    epoch: int,
    avg_losses: dict[str, float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float]],
    csv_path: Path,
    plot_path: Path,
    checkpoint_dir: Path,
):
    """Save loss row, update plot, and write checkpoint."""
    row = {"epoch": epoch + 1, **avg_losses}
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    plot_losses(csv_path, plot_path)

    ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        ckpt_path,
    )


def train_model(
    sound_dir: Path,
    labels_csv_path: Path,
    checkpoint_dir: Path,
    losses_dir: Path,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    device_id: int = 0,
    num_workers: int = 0,
    subset_indices: list[int] | None = None,
    dataset: PsychoAcousticDataset | None = None,
    audio_workers: int = 0,
    val_sound_dir: Path | None = None,
    val_dataset: PsychoAcousticDataset | None = None,
    use_scheduler: bool = True,
    statistics_dir: Path | None = None,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
) -> list[dict[str, float]]:
    print("=" * 100)
    device = _get_device(device_id)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = Path(losses_dir)
    losses_dir.mkdir(parents=True, exist_ok=True)
    csv_path = losses_dir / "losses.csv"
    plot_path = losses_dir / "losses.png"

    # ── Model ──
    if statistics_dir is None:
        statistics_dir = (
            Path(__file__).parent.parent
            / "data"
            / "standardized_audio_files"
            / "training_set"
            / "visualization"
        )
    statistics_dir = Path(statistics_dir)
    temporal_stats_path = (
        statistics_dir / "parameter_average_per_time_segment_train.csv"
    )
    value_stats_path = statistics_dir / "parameter_value_stats_train.csv"
    biases = _load_time_biases(temporal_stats_path)
    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
        )
        if use_scheduler
        else None
    )

    # ── Resume from latest checkpoint ──
    start_epoch, history = _resume_checkpoint(model, optimizer, checkpoint_dir)
    if not use_scheduler:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    if start_epoch == 0:
        csv_path.unlink(missing_ok=True)
        print("Starting from scratch")
        ckpt_path = checkpoint_dir / "epoch_0000.pt"
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": [],
            },
            ckpt_path,
        )
        print(f"Saved initial parameters to {ckpt_path.name}")
    elif history:
        rows = [{"epoch": i + 1, **h} for i, h in enumerate(history)]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Dataset ──
    if dataset is None:
        dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, subset_indices=subset_indices, audio_workers=audio_workers)
    if len(dataset) == 0:
        print("No data found — nothing to train on.")
        return []
    print(f"Loaded {len(dataset)} train audio-label pair(s)")

    # ── Validation dataset (physically separate folder, see split_train_val.py) ──
    if val_dataset is None and val_sound_dir is not None:
        val_dataset = PsychoAcousticDataset(val_sound_dir, labels_csv_path, audio_workers=audio_workers)
    if val_dataset is None or len(val_dataset) == 0:
        print("No validation data found — proceeding without validation.")
        val_dataset = None
    else:
        print(f"Loaded {len(val_dataset)} val audio-label pair(s)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=num_workers,
        )

    t_train_start = time.perf_counter()
    n_batches = len(loader)
    print(f"Training {epochs - start_epoch} epoch(s) ({n_batches} batches each)...")

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t_epoch = time.perf_counter()
        t_batch_start = time.perf_counter()
        epoch_losses: list[dict[str, float]] = []
        for batch_idx, (waveform, targets) in enumerate(loader):
            losses = _training_step(
                model,
                waveform,
                targets,
                optimizer,
                device,
                value_stats_path,
            )
            epoch_losses.append(losses)
            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{n_batches} ({time.perf_counter() - t_batch_start:.4f}s)")
                t_batch_start = time.perf_counter()
        t_total = time.perf_counter() - t_epoch

        avg_losses = {
            k: torch.tensor([b[k] for b in epoch_losses]).nanmean().item()
            for k in epoch_losses[0]
        }

        # ── Validation ──
        if val_loader is not None:
            model.eval()
            val_losses_epoch: list[dict[str, float]] = []
            with torch.no_grad():
                for waveform, targets in val_loader:
                    waveform = waveform.to(device)
                    targets = {n: t.to(device) for n, t in targets.items()}
                    preds = model(waveform)
                    trimmed = {n: targets[n][:, :preds[n].shape[-1]] for n in PARAM_NAMES}
                    v_losses = compute_loss(
                        model,
                        preds,
                        trimmed,
                        stats_path=value_stats_path,
                    )
                    val_losses_epoch.append({k: v.item() for k, v in v_losses.items()})
            model.train()

            avg_val_losses = {
                k: torch.tensor([b[k] for b in val_losses_epoch]).nanmean().item()
                for k in val_losses_epoch[0]
            }
            for k, v in avg_val_losses.items():
                avg_losses[f"val_{k}"] = v

            history.append(avg_losses)
            print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_losses['total']:.6f} — val_loss: {avg_losses['val_total']:.6f} — {t_total:.4f}s")
            if scheduler is not None:
                scheduler.step(avg_losses['val_total'])
        else:
            history.append(avg_losses)
            print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_losses['total']:.6f} — {t_total:.4f}s")
            if scheduler is not None:
                scheduler.step(avg_losses['total'])
        print(f"  current lr: {optimizer.param_groups[0]['lr']:.6f}")

        _log_epoch(epoch, avg_losses, model, optimizer, history, csv_path, plot_path, checkpoint_dir)

    return history
