from pathlib import Path

from data_preprocessing.calculate_reference_values import (
    calculate_reference_values,
    merge_reference_values,
)
from data_preprocessing.convert_to_wav import (
    convert_to_wav,
    split_reference_into_training_segments,
)
# from DL_model.train_model import PsychoAcousticDataset, run_comparison, train_model
# from tranings_data_visualization.visualize_data_distribution import main as calculate_biases


def main():
    root = Path("data") / "standardized_audio_files" / "training_set"
    raw_dir = Path("data") / "raw_audio_files"
    full_sound_dir = root / "full_sound_files"
    sound_dir = root / "sound_files"
    full_labels_dir = root / "full_recording_labels"
    one_second_labels_dir = root / "one_second_labels"
    labels_csv_path = root / "all_psychoacoustic_labels.csv"

    convert_to_wav(
        input_folder=raw_dir,
        output_folder=full_sound_dir,
        fs=48000,
        segment_length_s=None,
        number_samples=None
    )

    split_reference_into_training_segments(
        input_folder=full_sound_dir,
        output_folder=sound_dir,
        segment_length_s=1.0,
        include_partial_segment=False,
    )

    calculate_reference_values(
        input_folder=full_sound_dir,
        output_folder=full_labels_dir,
        max_workers=2,
    )

    calculate_reference_values(
        input_folder=sound_dir,
        output_folder=one_second_labels_dir,
        one_second=True,
        max_workers=12,
    )

    merge_reference_values(
        mapping_csv=sound_dir / "segment_mapping.csv",
        full_recording_labels=full_labels_dir,
        one_second_labels=one_second_labels_dir,
        output_csv=labels_csv_path,
    )

    # train_dir = sound_dir / "train"
    # val_dir = sound_dir / "val"
    # checkpoint_dir = Path("DL_model") / "epochs"
    # losses_dir = Path("DL_model") / "losses"
    #
    # calculate_biases()
    #
    # dataset = PsychoAcousticDataset(
    #     train_dir,
    #     labels_csv_path,
    #     audio_workers=12,
    # )
    # val_dataset = PsychoAcousticDataset(
    #     val_dir,
    #     labels_csv_path,
    #     audio_workers=12,
    # )
    #
    # train_model(
    #     sound_dir=train_dir,
    #     val_sound_dir=val_dir,
    #     val_dataset=val_dataset,
    #     labels_csv_path=labels_csv_path,
    #     checkpoint_dir=checkpoint_dir,
    #     losses_dir=losses_dir,
    #     epochs=100,
    #     lr=1e-3,
    #     batch_size=128,
    #     device_id=0,
    #     num_workers=0,
    #     use_scheduler=True,
    #     dataset=dataset,
    # )

  #Epoch 51/100 — loss: 50.812763 — val_loss: 51.131187 — 116.0714s
  # current lr: 0.000008
  # batch 100/483 (24.1882s)
  # batch 200/483 (24.0658s)
  # batch 300/483 (24.0510s)
  # batch 400/483 (24.0514s)

# ab 51 ohne 0.5 when stalled - lr = 0.001 constant

    # subset_indices = [3]
    # run_comparison(
    #     sound_dir=sound_dir,
    #     labels_csv_path=labels_csv_path,
    #     checkpoint_dir=checkpoint_dir,
    #     n_samples=1,
    #     device_id=0,
    #     subset_indices=subset_indices,
    #     epochs=[0, "newest"],
    #     dataset=dataset,
    #     n_benchmark=10000,
    # )



if __name__ == "__main__":
    main()

"""
Input: Sound File
Output: Psychoacoustic Parameters
Reference: Psychoacoustic Parameters berechnet via MOSQITO
"""
