"""Train the unchanged PsychoacousticModel on mapped one-second targets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .mapped_dataset import (
    MappedPsychoAcousticDataset,
    validate_recording_splits,
)
from .train_model import train_model


def run_training(arguments: argparse.Namespace) -> None:
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    validate_recording_splits(arguments.mapping_csv)
    common = {
        "mapping_csv": arguments.mapping_csv,
        "segment_dir": arguments.segment_dir,
        "targets_path": arguments.targets,
    }
    training = MappedPsychoAcousticDataset(**common, split="train")
    validation = MappedPsychoAcousticDataset(**common, split="val")

    run_dir = Path(arguments.output_dir) / arguments.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "PsychoacousticModel (unchanged)",
        "mapping_csv": str(Path(arguments.mapping_csv).resolve()),
        "segment_dir": str(Path(arguments.segment_dir).resolve()),
        "targets": str(Path(arguments.targets).resolve()),
        "statistics_dir": str(Path(arguments.statistics_dir).resolve()),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "epochs": arguments.epochs,
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "use_scheduler": not arguments.no_scheduler,
        "scheduler_patience": arguments.scheduler_patience,
        "scheduler_factor": arguments.scheduler_factor,
        "seed": arguments.seed,
    }
    with (run_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    print(json.dumps(config, indent=2))
    train_model(
        sound_dir=arguments.segment_dir,
        labels_csv_path=arguments.targets,
        checkpoint_dir=run_dir / "checkpoints",
        losses_dir=run_dir / "losses",
        epochs=arguments.epochs,
        lr=arguments.learning_rate,
        batch_size=arguments.batch_size,
        device_id=arguments.device_id,
        num_workers=arguments.num_workers,
        dataset=training,
        val_dataset=validation,
        use_scheduler=not arguments.no_scheduler,
        statistics_dir=arguments.statistics_dir,
        scheduler_patience=arguments.scheduler_patience,
        scheduler_factor=arguments.scheduler_factor,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(_parse_args())
