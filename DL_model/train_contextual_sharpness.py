"""Small training harness for the contextual-sharpness prototype.

This is intentionally a plumbing/overfitting experiment, not a trustworthy
generalization benchmark. A real validation split must group by reference
recording so neighboring seconds never cross between train and validation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .contextual_sharpness import (
    ContextualSharpnessDataset,
    HybridContextModel,
    collate_contextual_sharpness,
)


def _sharpness_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if not mask.any():
        raise ValueError("Batch has no finite sharpness targets")
    return ((predictions[mask] - targets[mask]) ** 2).mean()


def train_contextual_sharpness_prototype(
    dataset: ContextualSharpnessDataset,
    output_dir: Path,
    validation_dataset: ContextualSharpnessDataset | None = None,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: torch.device | None = None,
) -> pd.DataFrame:
    if len(dataset) < 2:
        raise ValueError("Prototype training requires at least two segments")
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if validation_dataset is None:
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices = indices[:validation_count]
        training_indices = indices[validation_count:]
        training_dataset = Subset(dataset, training_indices)
        validation_dataset_for_loader = Subset(dataset, validation_indices)
    else:
        group_column = (
            "recording_id" if "recording_id" in dataset.rows[0] else "reference_file"
        )
        training_references = {row[group_column] for row in dataset.rows}
        validation_references = {
            row[group_column] for row in validation_dataset.rows
        }
        overlap = training_references & validation_references
        if overlap:
            raise ValueError(
                f"Reference leakage between train and validation: {sorted(overlap)}"
            )
        training_indices = list(range(len(dataset)))
        validation_indices = list(range(len(validation_dataset)))
        training_dataset = dataset
        validation_dataset_for_loader = validation_dataset

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_contextual_sharpness,
    )
    validation_loader = DataLoader(
        validation_dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_contextual_sharpness,
    )

    model = HybridContextModel().to(device)
    for parameter in model.center_model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        model.sharpness_branch.parameters(),
        lr=learning_rate,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for center, context, targets in training_loader:
            center = center.to(device)
            context = context.to(device)
            target = targets["sharpness_din_tv"].to(device)
            prediction = model.forward_sharpness(center, context)
            loss = _sharpness_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.detach().cpu())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for center, context, targets in validation_loader:
                center = center.to(device)
                context = context.to(device)
                target = targets["sharpness_din_tv"].to(device)
                prediction = model.forward_sharpness(center, context)
                validation_losses.append(
                    _sharpness_loss(prediction, target).cpu()
                )

        row = {
            "epoch": epoch,
            "train_sharpness_mse": torch.stack(training_losses).mean().item(),
            "val_sharpness_mse": torch.stack(validation_losses).mean().item(),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: "
            f"train={row['train_sharpness_mse']:.6f}, "
            f"val={row['val_sharpness_mse']:.6f}"
        )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "losses.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "history": history,
        },
        output_dir / "prototype_checkpoint.pt",
    )
    return history_frame


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=data / "processed" / "1s_training" / "segment_mapping.csv",
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=data / "processed" / "1s_training",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=data / "processed" / "1file",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=data / "experiments" / "contextual_sharpness_prototype" / "hybrid_targets.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data / "experiments" / "contextual_sharpness_prototype" / "training",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--use-manifest-splits",
        action="store_true",
        help="Use train/val assignments from the mapping instead of a local random split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    common_dataset_arguments = {
        "mapping_csv": arguments.mapping_csv,
        "segment_dir": arguments.segment_dir,
        "reference_dir": arguments.reference_dir,
        "targets_path": arguments.targets,
    }
    prototype_dataset = ContextualSharpnessDataset(
        **common_dataset_arguments,
        split="train" if arguments.use_manifest_splits else None,
    )
    validation_dataset = (
        ContextualSharpnessDataset(
            **common_dataset_arguments,
            split="val",
        )
        if arguments.use_manifest_splits
        else None
    )
    train_contextual_sharpness_prototype(
        dataset=prototype_dataset,
        output_dir=arguments.output_dir,
        validation_dataset=validation_dataset,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
    )
