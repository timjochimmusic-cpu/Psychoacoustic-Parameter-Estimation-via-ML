from pathlib import Path

from data_preprocessing.calculate_reference_values import calculate_reference_values
from data_preprocessing.convert_to_wav import convert_to_wav
from DL_model.train_model import train_model, run_comparison
from DL_model.train_model import PsychoAcousticDataset
from training_data_visualization.merge_psychoacoustic_labels import merge_psychoacoustic_labels
import os
import sys

def main():
    data_dir = Path("data")

    raw_dir = data_dir / "raw" / "music_dataset"

    processed_dir = data_dir / "processed"
    sound_dir = processed_dir / "music_dataset_60s_converted"
    train_dir = sound_dir / "train"
    val_dir = sound_dir / "val"

    labels_dir = data_dir / "labels" / "music_dateset_60s_labels"
    labels_csv_path = (
        labels_dir / "music_dateset_60s_all_psychoacoustic_labels.csv"
    )

    checkpoint_dir = data_dir / "checkpoints"
    losses_dir = data_dir / "losses"

    # convert_to_wav(
    #     input_folder=raw_dir,
    #     output_folder=sound_dir,
    #     segment_length_s=None
    # )

    # calculate_reference_values(
    #     input_folder=sound_dir,
    #     output_folder=labels_dir
    # )

    # merge_psychoacoustic_labels(
    #     labels_dir= labels_dir
    # )

    len = 128
    len_val = round(0.2 * len)
    dataset = PsychoAcousticDataset(
        train_dir,
        labels_csv_path,
        # subset_indices=list(range(len)),
        audio_workers=12
    )

    val_dataset = PsychoAcousticDataset(
        val_dir, labels_csv_path,
        # subset_indices=list(range(len_val)),
        audio_workers=12
    )

    train_model(
        sound_dir=train_dir,
        val_sound_dir=val_dir,
        val_dataset=val_dataset,
        labels_csv_path=labels_csv_path,
        checkpoint_dir=checkpoint_dir,
        losses_dir=losses_dir,
        epochs=100,
        lr=1e-3,
        batch_size=128,
        device_id=0,
        num_workers=0,
        use_scheduler=True,
        dataset=dataset,
    )

  #Epoch 51/100 — loss: 50.812763 — val_loss: 51.131187 — 116.0714s
  # current lr: 0.000008
  # batch 100/483 (24.1882s)
  # batch 200/483 (24.0658s)
  # batch 300/483 (24.0510s)
  # batch 400/483 (24.0514s)

#     print("checkpoint_dir:", Path(checkpoint_dir).resolve())
#     print("epoch 0:", (Path(checkpoint_dir) / "epoch_0000.pt").resolve())
#     print("exists:", (Path(checkpoint_dir) / "epoch_0000.pt").exists())

# # ab 51 ohne 0.5 when stalled - lr = 0.001 constant

#     # subset_indices = [0]
#     subset_indices = [0]
#     run_comparison(
#         sound_dir=sound_dir,
#         labels_csv_path=labels_csv_path,
#         checkpoint_dir=checkpoint_dir,
#         n_samples=1,
#         device_id=0,
#         subset_indices=subset_indices,
#         epochs=[0, 60],
#         dataset=dataset,
#     )



if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        sys.exit(
            "This program must be started via Slurm (sbatch main.sh)."
        )

    main()