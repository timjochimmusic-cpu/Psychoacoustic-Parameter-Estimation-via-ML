import argparse
from pathlib import Path
import re

import pandas as pd
import librosa
import numpy as np
import soundfile as sf


def _recording_id(reference_stem: str) -> str:
    """Remove generated index, time-window, and channel suffixes."""
    recording = re.sub(r"^\[\d+\]_", "", reference_stem)
    recording = re.sub(
        r"_\d+-\d+ms(?:_ch\d+)?$",
        "",
        recording,
    )
    return recording


def convert_to_wav(
    input_folder: Path,
    output_folder: Path,
    fs: int = 48_000,
    segment_length_s: float | None = None,
    overlap_s: float = 0.0,
    number_samples: int | None = None,
    max_segments_per_source: int | None = None,
    include_partial_segment: bool = False,
):
    """
    Convert audio files to mono WAV files.

    Each channel is saved separately. If ``segment_length_s`` is given,
    the source audio is split into segments of that duration with
    ``overlap_s`` overlap.

    Parameters
    ----------
    input_folder : Path
        Folder containing the source audio files.
    output_folder : Path
        Folder in which the converted WAV files are stored.
    fs : int, optional
        Target sampling rate in Hz.
    segment_length_s : float | None, optional
        Segment duration in seconds. Use None to keep complete files.
    overlap_s : float, optional
        Overlap between adjacent segments in seconds.
    number_samples : int | None, optional
        Optional maximum number of output WAV files.
    max_segments_per_source : int | None, optional
        Optional maximum number of time segments retained from each original
        source file. Channel outputs do not count as separate time segments.
    include_partial_segment : bool, optional
        If True, also save the final incomplete segment.
    """
    print("=" * 100)

    if max_segments_per_source is not None and max_segments_per_source <= 0:
        raise ValueError("max_segments_per_source must be greater than zero")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_extensions = {
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".wma",
        ".wav",
    }

    if segment_length_s is not None:
        if segment_length_s <= 0:
            raise ValueError("segment_length_s must be greater than zero.")

        if overlap_s < 0:
            raise ValueError("overlap_s must not be negative.")

        if overlap_s >= segment_length_s:
            raise ValueError(
                "overlap_s must be shorter than segment_length_s."
            )

        segment_length_samples = round(segment_length_s * fs)
        overlap_samples = round(overlap_s * fs)
        hop_length_samples = segment_length_samples - overlap_samples

    audio_files = sorted(
        (
            path
            for path in input_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in audio_extensions
        ),
        key=lambda path: str(path).lower(),
    )

    if not audio_files:
        print(f"No audio files found under {input_folder}")
        return

    existing_stems: set[str] = set()
    max_idx = 0

    for output_file in output_folder.glob("*.wav"):
        stem = output_file.stem

        if "]" in stem:
            content_stem = stem.split("]", 1)[1].lstrip("_")
            existing_stems.add(content_stem)

            try:
                idx = int(stem.split("]", 1)[0].lstrip("["))
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
        else:
            existing_stems.add(stem)

    next_idx = max_idx + 1

    def enumerate_path(content_stem: str, idx: int) -> Path:
        return output_folder / f"[{idx:06d}]_{content_stem}.wav"

    for file_path in audio_files:
        try:
            audio, _ = librosa.load(
                file_path,
                sr=fs,
                mono=False,
            )

            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            n_channels, n_samples = audio.shape
            relative_stem = f"{file_path.parent.name}_{file_path.stem}"

            if segment_length_s is None:
                segments = [(0, n_samples)]

            else:
                segments: list[tuple[int, int]] = []

                if n_samples >= segment_length_samples:
                    for start_sample in range(
                        0,
                        n_samples - segment_length_samples + 1,
                        hop_length_samples,
                    ):
                        end_sample = start_sample + segment_length_samples
                        segments.append((start_sample, end_sample))

                if include_partial_segment:
                    if not segments:
                        segments.append((0, n_samples))
                    else:
                        next_start = segments[-1][0] + hop_length_samples

                        if next_start < n_samples:
                            segments.append((next_start, n_samples))

                if not segments:
                    print(
                        f"Skipping {file_path.name}: "
                        f"shorter than {segment_length_s:g} s"
                    )
                    continue

            if max_segments_per_source is not None:
                segments = segments[:max_segments_per_source]

            for start_sample, end_sample in segments:
                start_ms = round(start_sample / fs * 1000)
                end_ms = round(end_sample / fs * 1000)

                for channel_idx in range(n_channels):
                    channel_suffix = (
                        f"_ch{channel_idx + 1}"
                        if n_channels > 1
                        else ""
                    )

                    content_stem = (
                        f"{relative_stem}_"
                        f"{start_ms:05d}-{end_ms:05d}ms"
                        f"{channel_suffix}"
                    )

                    if content_stem in existing_stems:
                        print(
                            "Skipping (already generated): "
                            f"{content_stem}"
                        )
                        continue

                    if (
                        number_samples is not None
                        and next_idx > number_samples
                    ):
                        print(
                            f"Reached limit of {number_samples} files — "
                            "stopping."
                        )
                        return

                    output_path = enumerate_path(
                        content_stem,
                        next_idx,
                    )

                    audio_segment = audio[
                        channel_idx,
                        start_sample:end_sample,
                    ]

                    sf.write(
                        output_path,
                        audio_segment,
                        fs,
                    )

                    print(
                        f"Converted ({next_idx}): "
                        f"{file_path.name} -> {output_path.name}"
                    )

                    existing_stems.add(content_stem)
                    next_idx += 1

        except Exception as error:
            print(f"Failed: {file_path.name} -> {error}")


def split_reference_into_training_segments(
    input_folder: Path,
    output_folder: Path,
    segment_length_s: float = 1.0,
    include_partial_segment: bool = False,
):
    """
    Split reference WAV files into shorter training segments.

    The complete reference WAV stem is preserved in the training filename,
    followed by the time interval of the training segment within that
    reference file.

    Parameters
    ----------
    input_folder : Path
        Folder containing the reference WAV files.
    output_folder : Path
        Folder in which the training WAV files are stored.
    segment_length_s : float, optional
        Length of each training segment in seconds.
    include_partial_segment : bool, optional
        If True, also save a final segment shorter than ``segment_length_s``.
    """
    if segment_length_s <= 0:
        raise ValueError("segment_length_s must be greater than zero.")
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    mapping_rows = []

    reference_files = sorted(input_folder.glob("*.wav"))

    if not reference_files:
        print(f"No reference WAV files found under {input_folder}")
        return

    for reference_path in reference_files:
        audio, fs = sf.read(reference_path)

        if audio.ndim > 1:
            raise ValueError(
                f"{reference_path.name} is not mono. "
                "Reference files are expected to be mono."
            )

        segment_length_samples = round(segment_length_s * fs)
        n_samples = len(audio)

        if n_samples < segment_length_samples:
            if not include_partial_segment:
                print(
                    f"Skipping {reference_path.name}: "
                    f"shorter than {segment_length_s:g} s"
                )
                continue

            segments = [(0, n_samples)]

        else:
            segments = [
                (
                    start_sample,
                    start_sample + segment_length_samples,
                )
                for start_sample in range(
                    0,
                    n_samples - segment_length_samples + 1,
                    segment_length_samples,
                )
            ]

            if include_partial_segment:
                last_end = segments[-1][1]

                if last_end < n_samples:
                    segments.append((last_end, n_samples))

        for start_sample, end_sample in segments:
            start_ms = round(start_sample / fs * 1000)
            end_ms = round(end_sample / fs * 1000)

            segment = audio[start_sample:end_sample]

            training_stem = (
                f"{reference_path.stem}"
                f"__seg_{start_ms:05d}-{end_ms:05d}ms"
            )

            output_path = output_folder / f"{training_stem}.wav"

            if not output_path.exists():
                sf.write(
                    output_path,
                    segment,
                    fs,
                )

                print(
                    f"Created: {reference_path.name} "
                    f"-> {output_path.name}"
                )
            else:
                print(f"Skipping (already exists): {output_path.name}")

            mapping_rows.append({
                "segment_id": training_stem,
                "source_file": output_path.name,
                "reference_file": reference_path.name,
                "recording_id": _recording_id(reference_path.stem),
                "sample_rate": fs,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
            })

    mapping_path = output_folder / "segment_mapping.csv"

    pd.DataFrame(mapping_rows).to_csv(
        mapping_path,
        index=False,
    )

    print(f"Saved segment mapping: {mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert source audio to complete mono reference WAV files and "
            "optionally create one-second training segments."
        )
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("reference_folder", type=Path)
    parser.add_argument(
        "--training-segment-folder",
        type=Path,
        help=(
            "If provided, split the converted references into one-second "
            "training WAV files and write segment_mapping.csv."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    arguments = parser.parse_args()

    # Keep every source recording complete. Reference values can therefore be
    # calculated without artificial restarts at one- or 60-second boundaries.
    convert_to_wav(
        input_folder=arguments.input_folder,
        output_folder=arguments.reference_folder,
        fs=arguments.sample_rate,
        segment_length_s=None,
        overlap_s=0.0,
    )

    if arguments.training_segment_folder is not None:
        split_reference_into_training_segments(
            input_folder=arguments.reference_folder,
            output_folder=arguments.training_segment_folder,
            segment_length_s=1.0,
        )
