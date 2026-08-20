"""Controlled A/B training: isolated versus contextual sharpness."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .contextual_sharpness import (
    ContextualSharpnessBranch,
    ContextualSharpnessDataset,
)


class SharpnessComparisonDataset(Dataset):
    def __init__(
        self,
        contextual_dataset: ContextualSharpnessDataset,
        isolated_targets_path: Path,
    ):
        self.contextual_dataset = contextual_dataset
        payload = torch.load(
            isolated_targets_path,
            map_location="cpu",
            weights_only=True,
        )
        self.isolated_items = payload["items"]
        missing = set(contextual_dataset.stems) - set(self.isolated_items)
        if missing:
            raise ValueError(
                f"Missing isolated targets for {len(missing)} segment(s)"
            )

    def __len__(self) -> int:
        return len(self.contextual_dataset)

    def __getitem__(self, index: int):
        center, context, contextual_targets = self.contextual_dataset[index]
        stem = self.contextual_dataset.stems[index]
        isolated_target = self.isolated_items[stem]["target"]
        contextual_target = contextual_targets["sharpness_din_tv"]
        if isolated_target.numel() != contextual_target.numel():
            raise ValueError(
                f"{stem}: isolated/contextual frame counts differ "
                f"({isolated_target.numel()} vs {contextual_target.numel()})"
            )
        return center, context, isolated_target, contextual_target


def _collate(batch):
    return tuple(torch.stack([item[index] for item in batch]) for index in range(4))


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    return ((prediction[mask] - target[mask]) ** 2).mean()


def _train_mode(
    mode: str,
    training_dataset: SharpnessComparisonDataset,
    validation_dataset: SharpnessComparisonDataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[ContextualSharpnessBranch, pd.DataFrame]:
    if mode not in {"isolated", "contextual"}:
        raise ValueError(f"Unknown mode: {mode}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    model = ContextualSharpnessBranch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    checkpoint_path = output_dir / f"{mode}_best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for center, context, isolated, contextual in training_loader:
            center = center.to(device)
            context = context.to(device)
            target = isolated.to(device) if mode == "isolated" else contextual.to(device)
            waveform = center if mode == "isolated" else context
            prediction = model(waveform, center_samples=center.shape[-1])
            loss = _mse(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.detach().cpu())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for center, context, isolated, contextual in validation_loader:
                center = center.to(device)
                context = context.to(device)
                target = (
                    isolated.to(device)
                    if mode == "isolated"
                    else contextual.to(device)
                )
                waveform = center if mode == "isolated" else context
                prediction = model(waveform, center_samples=center.shape[-1])
                validation_losses.append(_mse(prediction, target).cpu())

        train_loss = torch.stack(training_losses).mean().item()
        validation_loss = torch.stack(validation_losses).mean().item()
        history.append(
            {
                "mode": mode,
                "epoch": epoch,
                "training_target_mse": train_loss,
                "validation_target_mse": validation_loss,
            }
        )
        print(
            f"{mode:10s} epoch {epoch:02d}/{epochs}: "
            f"train={train_loss:.6f}, val={validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "mode": mode,
                    "epoch": epoch,
                    "validation_target_mse": validation_loss,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"{mode}: selected epoch {best_epoch} with val={best_loss:.6f}")
    return model, pd.DataFrame(history)


def _contextual_metrics(
    model: ContextualSharpnessBranch,
    mode: str,
    dataset: SharpnessComparisonDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | str]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for center, context, _, contextual in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        ):
            center = center.to(device)
            waveform = center if mode == "isolated" else context.to(device)
            predictions.append(
                model(waveform, center_samples=center.shape[-1]).cpu()
            )
            targets.append(contextual)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = prediction - target
    return {
        "mode": mode,
        "contextual_mse": (error**2).mean().item(),
        "contextual_mae": error.abs().mean().item(),
        "contextual_correlation": float(
            np.corrcoef(prediction.flatten().numpy(), target.flatten().numpy())[0, 1]
        ),
    }


def _interval_evaluation(
    models: dict[str, ContextualSharpnessBranch],
    dataset: SharpnessComparisonDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: dict[str, list[torch.Tensor]] = {mode: [] for mode in models}
    isolated_targets = []
    contextual_targets = []
    with torch.no_grad():
        for center, context, isolated, contextual in DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        ):
            isolated_targets.append(isolated)
            contextual_targets.append(contextual)
            for mode, model in models.items():
                waveform = center if mode == "isolated" else context
                predictions[mode].append(
                    model(
                        waveform.to(device),
                        center_samples=center.shape[-1],
                    ).cpu()
                )

    isolated_target = torch.cat(isolated_targets)
    contextual_target = torch.cat(contextual_targets)
    predictions = {
        mode: torch.cat(parts) for mode, parts in predictions.items()
    }
    intervals_ms = ((0, 50), (50, 100), (100, 200), (200, 500), (500, 1000))
    model_rows = []
    label_rows = []
    frames_per_second = contextual_target.shape[-1]
    for start_ms, end_ms in intervals_ms:
        start_frame = round(start_ms / 1000 * frames_per_second)
        end_frame = round(end_ms / 1000 * frames_per_second)
        interval = slice(start_frame, end_frame)
        for mode, prediction in predictions.items():
            error = prediction[:, interval] - contextual_target[:, interval]
            model_rows.append(
                {
                    "interval_ms": f"{start_ms}-{end_ms}",
                    "mode": mode,
                    "contextual_mse": (error**2).mean().item(),
                    "contextual_mae": error.abs().mean().item(),
                }
            )
        label_error = (
            isolated_target[:, interval] - contextual_target[:, interval]
        )
        label_rows.append(
            {
                "interval_ms": f"{start_ms}-{end_ms}",
                "isolated_vs_contextual_mse": (label_error**2).mean().item(),
                "isolated_vs_contextual_mae": label_error.abs().mean().item(),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(label_rows)


def compare_training_modes(
    training_dataset: SharpnessComparisonDataset,
    validation_dataset: SharpnessComparisonDataset,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    histories = []
    results = []
    trained_models = {}
    for mode in ("isolated", "contextual"):
        model, history = _train_mode(
            mode=mode,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        histories.append(history)
        trained_models[mode] = model
        results.append(
            _contextual_metrics(
                model,
                mode,
                validation_dataset,
                batch_size,
                device,
            )
        )

    contextual_training_targets = torch.stack(
        [training_dataset[index][3] for index in range(len(training_dataset))]
    )
    contextual_validation_targets = torch.stack(
        [validation_dataset[index][3] for index in range(len(validation_dataset))]
    )
    mean_prediction = contextual_training_targets.mean(0).expand_as(
        contextual_validation_targets
    )
    baseline_error = mean_prediction - contextual_validation_targets
    results.append(
        {
            "mode": "mean_contextual_baseline",
            "contextual_mse": (baseline_error**2).mean().item(),
            "contextual_mae": baseline_error.abs().mean().item(),
            "contextual_correlation": float(
                np.corrcoef(
                    mean_prediction.flatten().numpy(),
                    contextual_validation_targets.flatten().numpy(),
                )[0, 1]
            ),
        }
    )

    pd.concat(histories, ignore_index=True).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "contextual_evaluation.csv", index=False)
    interval_frame, label_interval_frame = _interval_evaluation(
        trained_models,
        validation_dataset,
        batch_size,
        device,
    )
    interval_frame.to_csv(
        output_dir / "contextual_interval_evaluation.csv",
        index=False,
    )
    label_interval_frame.to_csv(
        output_dir / "label_interval_comparison.csv",
        index=False,
    )
    print("\nContextual-label evaluation:")
    print(result_frame.to_string(index=False))
    print("\nContextual-label evaluation by time interval:")
    print(interval_frame.to_string(index=False))
    print("\nRaw label disagreement by time interval:")
    print(label_interval_frame.to_string(index=False))
    return result_frame


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    experiment = root / "data" / "experiments" / "contextual_sharpness_multi_recording"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=experiment / "segments" / "segment_mapping.csv",
    )
    parser.add_argument("--segment-dir", type=Path, default=experiment / "segments")
    parser.add_argument("--reference-dir", type=Path, default=experiment / "references")
    parser.add_argument(
        "--contextual-targets",
        type=Path,
        default=experiment / "hybrid_targets.pt",
    )
    parser.add_argument(
        "--isolated-targets",
        type=Path,
        default=experiment / "isolated_sharpness_targets.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment / "ab_comparison",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    common = {
        "mapping_csv": arguments.mapping_csv,
        "segment_dir": arguments.segment_dir,
        "reference_dir": arguments.reference_dir,
        "targets_path": arguments.contextual_targets,
    }
    training = SharpnessComparisonDataset(
        ContextualSharpnessDataset(**common, split="train"),
        arguments.isolated_targets,
    )
    validation = SharpnessComparisonDataset(
        ContextualSharpnessDataset(**common, split="val"),
        arguments.isolated_targets,
    )
    compare_training_modes(
        training_dataset=training,
        validation_dataset=validation,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
