import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from mosqito import (
    loudness_zwtv,
    sharpness_din_tv,
    roughness_dw,
    tnr_ecma_perseg,
    sii_ansi,
)


RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


FULL_RECORDING_PARAM_CONFIGS = [
    (
        "loudness_zwtv",
        loudness_zwtv,
        (),
        {},
    ),
    (
        "sharpness_din_tv",
        sharpness_din_tv,
        (),
        {},
    ),
]

ONE_SECOND_PARAM_CONFIGS = [
    (
        "roughness_dw",
        roughness_dw,
        (),
        {},
    ),
    (
        "tnr_ecma_perseg",
        tnr_ecma_perseg,
        (),
        {},
    ),
    (
        "sii_ansi",
        sii_ansi,
        ("critical", "normal"),
        {},
    ),
]

PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]

FRAME_COUNTS = {
    "loudness_zwtv": 500,
    "sharpness_din_tv": 500,
    "roughness_dw": 9,
    "tnr_ecma_perseg": 2,
    "sii_ansi": 1,
}


def calculate_reference_values(
    input_folder: Path,
    output_folder: Path,
    one_second: bool = False,
    max_workers: int = 12,
):
    """
    Compute psychoacoustic reference values for mono WAV files.

    For time-dependent parameters, both the parameter values and the
    corresponding time axis returned by MoSQITo are stored.

    Different psychoacoustic parameters may have different temporal
    resolutions. Therefore, every parameter receives its own time column.

    By default, this stage calculates Loudness and Sharpness on complete
    recordings. With ``one_second=True``, it instead calculates Roughness,
    TNR, and SII on isolated one-second training segments.

    Shorter columns are padded with NaN only for CSV storage. The DataFrame
    row index has no temporal meaning across different parameters.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    print("=" * 100)
    sys.stdout = _ColorStdout(sys.stdout)

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_paths = sorted(
        input_folder.glob("*.wav")
    )

    parameter_configs = (
        ONE_SECOND_PARAM_CONFIGS
        if one_second
        else FULL_RECORDING_PARAM_CONFIGS
    )

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_info = {}
        submitted_count = {}

        for audio_path in audio_paths:

            name = audio_path.stem

            output_file = (
                output_folder
                / f"{name}.csv"
            )

            if output_file.exists():
                print(
                    f"Skipping (already exists): "
                    f"{output_file.name}"
                )
                continue

            submitted_count[name] = 0

            for (
                param_name,
                fn,
                args,
                kwargs,
            ) in parameter_configs:

                future = executor.submit(
                    _compute_param,
                    name,
                    param_name,
                    fn,
                    str(audio_path),
                    *args,
                    **kwargs,
                )

                future_to_info[future] = (
                    name,
                    param_name,
                )

                submitted_count[name] += 1

        file_results = {}
        remaining = {}

        for future in as_completed(
            future_to_info
        ):
            name, param_name = (
                future_to_info[future]
            )

            try:
                result = future.result()

                (
                    _,
                    _,
                    values,
                    times,
                    error,
                ) = result

                file_results.setdefault(
                    name,
                    {},
                )[param_name] = {
                    "values": values,
                    "times": times,
                }

                if error:
                    print(
                        f"{RED}Failed "
                        f"{name}/{param_name}: "
                        f"{error}{RESET}"
                    )

            except Exception as exc:

                file_results.setdefault(
                    name,
                    {},
                )[param_name] = {
                    "values": None,
                    "times": None,
                }

                print(
                    f"{RED}Failed "
                    f"{name}/{param_name}: "
                    f"{exc}{RESET}"
                )

            rem = (
                remaining.get(
                    name,
                    submitted_count[name],
                )
                - 1
            )

            remaining[name] = rem

            if rem == 0:

                params = file_results.pop(
                    name
                )

                remaining.pop(name)

                df = _build_reference_dataframe(
                    params
                )

                output_file = (
                    output_folder
                    / f"{name}.csv"
                )

                df.to_csv(
                    output_file,
                    index=False,
                )

                print(
                    f"{GREEN}Saved: "
                    f"{output_file}{RESET}"
                )


def _compute_param(
    name,
    param_name,
    fn,
    audio_path_str,
    *args,
    **kwargs,
):
    """
    Compute one psychoacoustic parameter.

    Returns
    -------
    values : np.ndarray
        Psychoacoustic parameter trajectory or scalar.

    times : np.ndarray | None
        Time positions returned by MoSQITo for time-dependent parameters.
        None for scalar/global parameters such as SII.
    """

    audio_path = Path(
        audio_path_str
    )

    signal, sr = sf.read(
        audio_path
    )

    duration_s = (
        len(signal) / sr
    )

    np.seterr(
        invalid="ignore",
        divide="ignore",
    )

    t0 = time.perf_counter()

    try:
        result = fn(
            signal,
            sr,
            *args,
            **kwargs,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        print(
            f"[{name}] "
            f"{fn.__name__}: "
            f"{elapsed:.4f}s"
        )

        values, times = (
            _extract_values_and_time(
                param_name=param_name,
                result=result,
                signal_duration_s=duration_s,
            )
        )

        return (
            name,
            param_name,
            values,
            times,
            None,
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - t0
        )

        print(
            f"{RED}[{name}] "
            f"{fn.__name__}: FAILED "
            f"({exc}) - "
            f"duration {duration_s:.2f}s"
            f"{RESET}",
            flush=True,
        )

        return (
            name,
            param_name,
            None,
            None,
            str(exc),
        )


def _extract_values_and_time(
    param_name: str,
    result,
    signal_duration_s: float,
):
    """
    Extract the main parameter values and their time axis from a MoSQITo
    return value.

    The first returned object is assumed to contain the requested metric,
    matching the current use of the MoSQITo functions.

    For a time-dependent parameter, a matching 1-D time axis is searched
    among the additional objects returned by MoSQITo.

    Importantly, no time axis is reconstructed from the signal duration or
    from the number of parameter values.
    """

    if isinstance(
        result,
        tuple,
    ):
        outputs = result
    else:
        outputs = (result,)

    values = np.asarray(
        outputs[0],
        dtype=float,
    )

    if values.ndim == 0:
        values = values.reshape(1)

    # Support scalar metrics when this helper is reused by another stage.
    if values.size == 1:
        return (
            values.reshape(-1),
            None,
        )

    # The metric itself should be one-dimensional.
    if values.ndim != 1:
        raise ValueError(
            f"{param_name}: expected the first "
            f"MoSQITo output to be 1-D, "
            f"got shape {values.shape}"
        )

    time_candidates = []

    for output in outputs[1:]:

        try:
            candidate = np.asarray(
                output,
                dtype=float,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if candidate.ndim != 1:
            continue

        if len(candidate) != len(values):
            continue

        if len(candidate) < 2:
            continue

        finite = candidate[
            np.isfinite(candidate)
        ]

        if len(finite) != len(candidate):
            continue

        # A time axis must be monotonically increasing.
        if not np.all(
            np.diff(candidate) >= 0
        ):
            continue

        # Time positions should lie within the reference WAV.
        tolerance_s = 0.1

        if candidate[0] < -tolerance_s:
            continue

        if (
            candidate[-1]
            > signal_duration_s
            + tolerance_s
        ):
            continue

        time_candidates.append(
            candidate
        )

    if len(time_candidates) == 0:
        raise ValueError(
            f"{param_name}: MoSQITo returned "
            f"{len(outputs)} output object(s), "
            "but no matching time axis could "
            "be identified. "
            "No artificial time axis will be "
            "generated."
        )

    if len(time_candidates) > 1:
        raise ValueError(
            f"{param_name}: more than one "
            "possible time axis was found. "
            "Please inspect the MoSQITo return "
            "values instead of guessing."
        )

    times = time_candidates[0]

    return (
        values,
        times,
    )


def _build_reference_dataframe(
    params: dict,
) -> pd.DataFrame:
    """
    Build one reference-label DataFrame.

    Each time-dependent parameter has its own time column.

    NaN padding is used solely because pandas requires all DataFrame columns
    to have the same length. Padding does not imply temporal alignment
    between rows of different parameters.
    """

    columns = {}

    for (
        param_name,
        result,
    ) in params.items():

        values = result["values"]
        times = result["times"]

        if values is None:
            values = np.array(
                [np.nan],
                dtype=float,
            )

        values = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        if times is not None:

            times = np.asarray(
                times,
                dtype=float,
            ).reshape(-1)

            if len(times) != len(values):
                raise ValueError(
                    f"{param_name}: "
                    f"{len(values)} values but "
                    f"{len(times)} time positions."
                )

            columns[
                f"{param_name}_time_s"
            ] = times

        columns[param_name] = values

    max_len = max(
        len(column)
        for column
        in columns.values()
    )

    padded_columns = {
        name: _pad(
            values,
            max_len,
        )
        for (
            name,
            values,
        ) in columns.items()
    }

    return pd.DataFrame(
        padded_columns
    )


def _pad(arr, target_len):
    arr = np.asarray(arr, dtype=float)

    if len(arr) >= target_len:
        return arr[:target_len]

    return np.pad(
        arr,
        (0, target_len - len(arr)),
        constant_values=np.nan,
    )


def _load_parameter(
    frame: pd.DataFrame,
    parameter_name: str,
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    if parameter_name not in frame:
        raise ValueError(f"{csv_path} is missing {parameter_name}")

    time_column = f"{parameter_name}_time_s"
    if time_column not in frame:
        values = frame[parameter_name].dropna().to_numpy(dtype=float)
        return values, None

    valid = frame[[parameter_name, time_column]].dropna()
    values = valid[parameter_name].to_numpy(dtype=float)
    times = valid[time_column].to_numpy(dtype=float)
    return values, times


def _slice_full_recording_parameter(
    values: np.ndarray,
    times: np.ndarray,
    start_s: float,
    frame_count: int,
) -> np.ndarray:
    """Map one second of a full trajectory to the model's output positions."""
    result = np.full(frame_count, np.nan, dtype=float)
    if len(times) == 0:
        return result

    frame_period_s = 1.0 / frame_count
    tolerance_s = frame_period_s / 4
    left_aligned = abs(times[0]) <= tolerance_s
    relative_times = times - start_s

    if left_aligned:
        positions = np.rint(relative_times / frame_period_s).astype(int)
        expected_times = start_s + positions * frame_period_s
    else:
        positions = np.rint(relative_times / frame_period_s).astype(int) - 1
        expected_times = start_s + (positions + 1) * frame_period_s

    selected = (positions >= 0) & (positions < frame_count)
    selected_positions = positions[selected]
    if len(np.unique(selected_positions)) != len(selected_positions):
        raise ValueError("Multiple reference values map to the same frame")
    if np.any(np.abs(times[selected] - expected_times[selected]) > tolerance_s):
        raise ValueError("Reference timestamps do not match the 500-frame grid")

    result[selected_positions] = values[selected]
    return result


def merge_reference_values(
    mapping_csv: Path,
    full_recording_labels: Path,
    one_second_labels: Path,
    output_csv: Path,
):
    """Combine mixed-context references into the existing training CSV format."""
    mapping = pd.read_csv(mapping_csv)
    required_columns = {
        "segment_id",
        "reference_file",
        "recording_id",
        "sample_rate",
        "start_sample",
        "start_ms",
        "end_ms",
        "duration_ms",
    }
    missing = required_columns.difference(mapping.columns)
    if missing:
        raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
    if mapping["segment_id"].duplicated().any():
        raise ValueError("Mapping contains duplicate segment_id values")
    if not (mapping["duration_ms"] == 1000).all():
        raise ValueError("Only complete one-second segments can be merged")
    if output_csv.exists():
        raise FileExistsError(f"Output already exists: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_csv.with_suffix(output_csv.suffix + ".tmp")
    full_label_cache = {}
    fieldnames = [
        "source_file",
        "segment_id",
        "recording_id",
        "reference_file",
        "start_ms",
        "end_ms",
        "time_index",
        *PARAM_NAMES,
    ]

    with temporary_output.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for completed, row in enumerate(mapping.itertuples(index=False), start=1):
            reference_name = str(row.reference_file)
            if reference_name not in full_label_cache:
                reference_csv = (
                    full_recording_labels / f"{Path(reference_name).stem}.csv"
                )
                full_frame = pd.read_csv(reference_csv)
                full_label_cache[reference_name] = {
                    parameter_name: _load_parameter(
                        full_frame,
                        parameter_name,
                        reference_csv,
                    )
                    for parameter_name in (
                        "loudness_zwtv",
                        "sharpness_din_tv",
                    )
                }

            targets = {}
            start_s = float(row.start_sample) / int(row.sample_rate)
            for parameter_name in ("loudness_zwtv", "sharpness_din_tv"):
                values, times = full_label_cache[reference_name][parameter_name]
                if times is None:
                    raise ValueError(
                        f"Full-recording {parameter_name} has no time axis"
                    )
                targets[parameter_name] = _slice_full_recording_parameter(
                    values,
                    times,
                    start_s,
                    FRAME_COUNTS[parameter_name],
                )

            segment_label_csv = one_second_labels / f"{row.segment_id}.csv"
            segment_frame = pd.read_csv(segment_label_csv)
            for parameter_name in ("roughness_dw", "tnr_ecma_perseg", "sii_ansi"):
                values, _ = _load_parameter(
                    segment_frame,
                    parameter_name,
                    segment_label_csv,
                )
                expected_count = FRAME_COUNTS[parameter_name]
                if len(values) not in (0, expected_count):
                    raise ValueError(
                        f"{segment_label_csv}/{parameter_name}: expected "
                        f"{expected_count} values, got {len(values)}"
                    )
                targets[parameter_name] = _pad(values, expected_count)

            for time_index in range(500):
                output_row = {
                    "source_file": f"{row.segment_id}.csv",
                    "segment_id": row.segment_id,
                    "recording_id": row.recording_id,
                    "reference_file": reference_name,
                    "start_ms": int(row.start_ms),
                    "end_ms": int(row.end_ms),
                    "time_index": time_index,
                }
                for parameter_name in PARAM_NAMES:
                    parameter_values = targets[parameter_name]
                    output_row[parameter_name] = (
                        parameter_values[time_index]
                        if time_index < len(parameter_values)
                        else ""
                    )
                writer.writerow(output_row)

            print(f"Merged {completed}/{len(mapping)}: {row.segment_id}")

    temporary_output.replace(output_csv)
    print(f"Saved combined training labels: {output_csv}")

class _ColorStdout:
    def __init__(
        self,
        original,
    ):
        self.original = original

    def write(
        self,
        text,
    ):
        if text.startswith(
            "[Warning]"
        ):
            text = (
                f"{BLUE}"
                f"{text}"
                f"{RESET}"
            )

        self.original.write(
            text
        )

    def flush(
        self,
    ):
        self.original.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calculate psychoacoustic references or merge previously "
            "calculated references into a training CSV."
        )
    )
    parser.add_argument("input_folder", type=Path, nargs="?")
    parser.add_argument("output_folder", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--one-second",
        action="store_true",
        help=(
            "Calculate Roughness, TNR, and SII for isolated one-second "
            "training files instead of full-recording Loudness/Sharpness."
        ),
    )
    parser.add_argument(
        "--merge",
        nargs=4,
        type=Path,
        metavar=(
            "MAPPING_CSV",
            "FULL_LABEL_DIR",
            "ONE_SECOND_LABEL_DIR",
            "OUTPUT_CSV",
        ),
        help="Merge both calculation modes into one training-compatible CSV.",
    )
    arguments = parser.parse_args()

    if arguments.merge is not None:
        if arguments.input_folder is not None or arguments.output_folder is not None:
            parser.error("input_folder/output_folder cannot be used with --merge")
        merge_reference_values(*arguments.merge)
    else:
        if arguments.input_folder is None or arguments.output_folder is None:
            parser.error("input_folder and output_folder are required")
        calculate_reference_values(
            input_folder=arguments.input_folder,
            output_folder=arguments.output_folder,
            one_second=arguments.one_second,
            max_workers=arguments.workers,
        )
