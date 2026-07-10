"""
Splits the audio files in training set and validation set and puts them in two different folders.

"""


import random
import shutil
from pathlib import Path
 
 
def split_train_val(
    sound_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    sound_dir = Path(sound_dir)
    train_dir = sound_dir / "train"
    val_dir = sound_dir / "val"
 
    # Falls der Split schon existiert, nicht neu würfeln — sonst würden sich
    # train/val bei jedem erneuten Lauf verschieben.
    if train_dir.exists() and val_dir.exists() and any(train_dir.glob("*.wav")):
        n_train = len(list(train_dir.glob("*.wav")))
        n_val = len(list(val_dir.glob("*.wav")))
        print(f"Split already exists: {n_train} train / {n_val} val — skipping")
        return train_dir, val_dir
 
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
 
    wav_files = sorted(sound_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found directly in {sound_dir} — nothing to split")
        return train_dir, val_dir
 
    rng = random.Random(seed)
    shuffled = wav_files.copy()
    rng.shuffle(shuffled)
 
    n_val = max(1, int(len(shuffled) * val_split))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]
 
    print(f"Moving {len(wav_files)} files: {len(train_files)} train / {len(val_files)} val")
 
    for f in train_files:
        shutil.move(str(f), str(train_dir / f.name))
    for f in val_files:
        shutil.move(str(f), str(val_dir / f.name))
 
    print(f"Done. train_dir={train_dir}  val_dir={val_dir}")
    print(f"Note: original files have been moved out of {sound_dir} (not duplicated).")
    return train_dir, val_dir
 
 
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data" / "standardized_audio_files" / "training_set"
    sound_dir = root / "sound_files"
 
    split_train_val(sound_dir, val_split=0.2, seed=42)