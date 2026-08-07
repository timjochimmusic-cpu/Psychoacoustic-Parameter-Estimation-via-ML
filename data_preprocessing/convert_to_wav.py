from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SHARPNESS_FRAME_S = 0.002
SHARPNESS_OVERLAP_S = SHARPNESS_FRAME_S * 5  # 0.01 s

def convert_to_wav(

    input_folder: Path,

    output_folder: Path,

    fs: int = 48_000,

    segment_length_s: float | None = 60.0,

    overlap_s: float = SHARPNESS_OVERLAP_S,

    number_samples: int | None = None,

    include_partial_segment: bool = False,

):

    """

    Convert audio files to mono WAV files.

    Each channel is saved separately.

    If segment_length_s is a number, files are divided into segments of that

    duration with overlap_s overlap.

    If segment_length_s is None, each complete input file is written without

    segmentation.

    Parameters

    ----------

    input_folder:

        Folder containing the source audio files.

    output_folder:

        Folder in which the converted WAV files are stored.

    fs:

        Target sampling rate in Hz.

    segment_length_s:

        Segment duration in seconds. Use None to keep complete files.

    overlap_s:

        Overlap between adjacent segments in seconds. Ignored when

        segment_length_s is None.

    number_samples:

        Optional maximum number of output WAV files.

    include_partial_segment:

        If True, also save the final segment when it is shorter than

        segment_length_s. Ignored when segment_length_s is None.

    """

    print("=" * 100)

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

                # Use the complete file as one segment.

                segments = [(0, n_samples)]

            else:

                segments: list[tuple[int, int]] = []

                # Add all complete segments.

                if n_samples >= segment_length_samples:

                    for start_sample in range(

                        0,

                        n_samples - segment_length_samples + 1,

                        hop_length_samples,

                    ):

                        end_sample = start_sample + segment_length_samples

                        segments.append((start_sample, end_sample))

                # Optionally add one final incomplete segment.

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
# def convert_to_wav(
#     input_folder: Path,
#     output_folder: Path,
#     fs: int = 48_000,
#     segment_length_s: float = 60.0,
#     overlap_s: float = SHARPNESS_OVERLAP_S,
#     number_samples: int | None = None,
#     include_partial_segment: bool = False,
# ):
#     """
#     Convert audio files to mono WAV segments.

#     Each channel is saved separately. Files longer than segment_length_s are
#     divided into segments with overlap_s overlap.

#     Parameters
#     ----------
#     input_folder:
#         Folder containing the source audio files.
#     output_folder:
#         Folder in which the converted WAV files are stored.
#     fs:
#         Target sampling rate in Hz.
#     segment_length_s:
#         Segment duration in seconds.
#     overlap_s:
#         Overlap between adjacent segments in seconds.
#     number_samples:
#         Optional maximum number of output WAV files.
#     include_partial_segment:
#         If True, also save the final segment when it is shorter than
#         segment_length_s.
#     """
#     print("=" * 100)

#     input_folder = Path(input_folder)
#     output_folder = Path(output_folder)
#     output_folder.mkdir(parents=True, exist_ok=True)

#     if segment_length_s <= 0:
#         raise ValueError("segment_length_s must be greater than zero.")

#     if overlap_s < 0:
#         raise ValueError("overlap_s must not be negative.")

#     if overlap_s >= segment_length_s:
#         raise ValueError("overlap_s must be shorter than segment_length_s.")

#     audio_extensions = {
#         ".mp3",
#         ".flac",
#         ".m4a",
#         ".aac",
#         ".ogg",
#         ".wma",
#         ".wav",
#     }

#     segment_length_samples = round(segment_length_s * fs)
#     overlap_samples = round(overlap_s * fs)
#     hop_length_samples = segment_length_samples - overlap_samples

#     audio_files = sorted(
#         (
#             path
#             for path in input_folder.rglob("*")
#             if path.is_file()
#             and path.suffix.lower() in audio_extensions
#         ),
#         key=lambda path: str(path).lower(),
#     )

#     if not audio_files:
#         print(f"No audio files found under {input_folder}")
#         return

#     existing_stems: set[str] = set()
#     max_idx = 0

#     for output_file in output_folder.glob("*.wav"):
#         stem = output_file.stem

#         if "]" in stem:
#             content_stem = stem.split("]", 1)[1].lstrip("_")
#             existing_stems.add(content_stem)

#             try:
#                 idx = int(stem.split("]", 1)[0].lstrip("["))
#                 max_idx = max(max_idx, idx)
#             except ValueError:
#                 pass
#         else:
#             existing_stems.add(stem)

#     next_idx = max_idx + 1

#     def enumerate_path(content_stem: str, idx: int) -> Path:
#         return output_folder / f"[{idx:06d}]_{content_stem}.wav"

#     for file_path in audio_files:
#         try:
#             audio, _ = librosa.load(
#                 file_path,
#                 sr=fs,
#                 mono=False,
#             )

#             if audio.ndim == 1:
#                 audio = audio[np.newaxis, :]

#             n_channels, n_samples = audio.shape
#             relative_stem = f"{file_path.parent.name}_{file_path.stem}"

#             # Start positions of all complete segments.
#             if n_samples >= segment_length_samples:
#                 start_samples = list(
#                     range(
#                         0,
#                         n_samples - segment_length_samples + 1,
#                         hop_length_samples,
#                     )
#                 )
#             else:
#                 start_samples = []

#             # Optionally include the final incomplete segment.
#             if include_partial_segment:
#                 if not start_samples:
#                     start_samples = [0]
#                 else:
#                     next_start = start_samples[-1] + hop_length_samples

#                     if next_start < n_samples:
#                         start_samples.append(next_start)

#             if not start_samples:
#                 print(
#                     f"Skipping {file_path.name}: "
#                     f"shorter than {segment_length_s:g} s"
#                 )
#                 continue

#             for start_sample in start_samples:
#                 end_sample = min(
#                     start_sample + segment_length_samples,
#                     n_samples,
#                 )

#                 start_ms = round(start_sample / fs * 1000)
#                 end_ms = round(end_sample / fs * 1000)

#                 for channel_idx in range(n_channels):
#                     channel_suffix = (
#                         f"_ch{channel_idx + 1}"
#                         if n_channels > 1
#                         else ""
#                     )

#                     content_stem = (
#                         f"{relative_stem}_"
#                         f"{start_ms:05d}-{end_ms:05d}ms"
#                         f"{channel_suffix}"
#                     )

#                     if content_stem in existing_stems:
#                         print(
#                             "Skipping (already generated): "
#                             f"{content_stem}"
#                         )
#                         continue

#                     if (
#                         number_samples is not None
#                         and next_idx > number_samples
#                     ):
#                         print(
#                             f"Reached limit of {number_samples} files — "
#                             "stopping."
#                         )
#                         return

#                     output_path = enumerate_path(
#                         content_stem,
#                         next_idx,
#                     )

#                     audio_segment = audio[
#                         channel_idx,
#                         start_sample:end_sample,
#                     ]

#                     sf.write(
#                         output_path,
#                         audio_segment,
#                         fs,
#                     )

#                     print(
#                         f"Converted ({next_idx}): "
#                         f"{file_path.name} -> {output_path.name}"
#                     )

#                     existing_stems.add(content_stem)
#                     next_idx += 1

#         except Exception as error:
#             print(f"Failed: {file_path.name} -> {error}")

# from pathlib import Path
# import librosa
# import soundfile as sf
# import numpy as np



# SHARPNESS_FRAME_S = 0.002
# SHARPNESS_OVERLAP_S = SHARPNESS_FRAME_S * 5

# def convert_to_wav(input_folder: Path, output_folder: Path, fs=48000, win_length_samples=5, number_samples=None):
#     """
#     Convert all audio files in input_folder to WAV format
#     with sampling rate fs and save them in output_folder.
#     Split into 5s windows with 0.01s overlap (5 sharpness frames)
#     to ensure full coverage after the sharpness skip.
#     Skip already generated audio files.
#     If number_samples is set, stop after that many total files exist in the output folder.
#     """
#     print("=" * 100)
#     output_folder.mkdir(parents=True, exist_ok=True)

#     audio_extensions = {
#         ".mp3", ".flac", ".m4a", ".aac",
#         ".ogg", ".wma", ".wav"
#     }

#     win_length_samples = int(win_length_samples * fs)
#     sharpness_length_samples = int(SHARPNESS_FRAME_S * fs)
#     hop_length = win_length_samples - sharpness_length_samples

#     extensions_lower = {ext.lower() for ext in audio_extensions}
#     audio_files = [
#         p for p in input_folder.rglob("*")
#         if p.is_file() and p.suffix.lower() in extensions_lower
#     ]
#     if not audio_files:
#         print(f"No audio files found under {input_folder}")
#         return

#     existing_stems: set[str] = set()
#     max_idx = 0
#     for f in output_folder.glob("*.wav"):
#         stem = f.stem
#         if "]" in stem:
#             content = stem.split("]", 1)[1].lstrip("_")
#             existing_stems.add(content)
#             try:
#                 idx = int(stem.split("]")[0].lstrip("["))
#                 max_idx = max(max_idx, idx)
#             except ValueError:
#                 pass
#         else:
#             existing_stems.add(stem)

#     next_idx = max_idx + 1

#     def _enumerate_path(base_stem: str, idx: int, ch_suffix: str = "") -> Path:
#         return output_folder / f"[{idx:06d}]_{base_stem}{ch_suffix}.wav"

#     def _check_limit(generated_count):
#         if number_samples is not None and generated_count > number_samples:
#             print(f"Reached limit of {number_samples} files — stopping.")
#             return True
#         return False

#     for file_path in audio_files:

#         try:
#             audio, _ = librosa.load(file_path, sr=fs, mono=False)

#             if audio.ndim == 1:
#                 audio = audio[np.newaxis, :]

#             n_channels, n_samples = audio.shape
#             rel_stem = f"{file_path.parent.name}_{file_path.stem}"

#             duration_ms = int(n_samples / fs * 1000)

#             for ch_idx in range(n_channels):
#                 ch_suffix = f"_ch{ch_idx + 1}" if n_channels > 1 else ""
#                 content_stem = f"{rel_stem}_00000-{duration_ms:05d}ms{ch_suffix}"

#                 if content_stem in existing_stems:
#                     print(f"Skipping (already generated): {content_stem}")
#                     continue

#                 idx = next_idx
#                 if _check_limit(idx):
#                     return

#                 output_path = _enumerate_path(content_stem, idx)
#                 sf.write(output_path, audio[ch_idx], fs)

#                 existing_stems.add(content_stem)
#                 next_idx = idx + 1
#             print(f"Converted ({idx}): {file_path.name} -> {output_path.name}")

#         except Exception as e:
#             print(f"Failed: {file_path.name} -> {e}")

    
