from pathlib import Path

from data_preprocessing.calculate_reference_values import calculate_reference_values
from data_preprocessing.convert_to_wav import convert_to_wav
from DL_model.train_model import train_model, run_comparison
from DL_model.train_model import PsychoAcousticDataset


def main():
    root = Path("data") / "standardized_audio_files" / "training_set"
    raw_dir = Path("data") / "raw_audio_files"
    sound_dir = root / "sound_files"
    train_dir = sound_dir / "train"    
    val_dir = sound_dir / "val"
    labels_csv_path = root / "all_psychoacoustic_labels.csv"
    checkpoint_dir = Path("DL_model") / "epochs"
    losses_dir = Path("DL_model") / "losses"
    labels_dir = Path("data") / "reference_data_full_file"

    convert_to_wav(
        input_folder=raw_dir,
        output_folder=sound_dir,
        fs=48000,
        win_length_samples=1,
        number_samples=None
    )

    calculate_reference_values(
        input_folder=sound_dir,
        output_folder=labels_dir
    )

    # len = 128
    # len_val = round(0.2 * len)
    # dataset = PsychoAcousticDataset(
    #     train_dir,
    #     labels_csv_path,
    #     # subset_indices=list(range(len)),
    #     audio_workers=12
    # )

    # val_dataset = PsychoAcousticDataset(
    #     val_dir, labels_csv_path,
    #     # subset_indices=list(range(len_val)),
    #     audio_workers=12
    # )

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
    main()

"""
Input: Sound File
Output: Psychoacoustic Parameters
Reference: Psychoacoustic Parameters berechnet via MOSQITO
"""