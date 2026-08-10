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


PARAM_CONFIGS = [
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


def calculate_reference_values(
    input_folder: Path,
    output_folder: Path,
):
    """
    Compute psychoacoustic reference values for mono WAV files.

    For time-dependent parameters, both the parameter values and the
    corresponding time axis returned by MoSQITo are stored.

    Different psychoacoustic parameters may have different temporal
    resolutions. Therefore, every parameter receives its own time column.

    Shorter columns are padded with NaN only for CSV storage. The DataFrame
    row index has no temporal meaning across different parameters.
    """

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

    with ProcessPoolExecutor(
        max_workers=12
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
            ) in PARAM_CONFIGS:

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

    # Global parameter, currently SII.
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

    calculate_reference_values(
        input_folder=Path(
            "/Users/hassi/Projects/"
            "Psychoacoustic-Parameter-Estimation-via-ML/"
            "data/processed/1file"
        ),
        output_folder=Path(
            "/Users/hassi/Projects/"
            "Psychoacoustic-Parameter-Estimation-via-ML/"
            "data/labels/test_1file"
        ),
    )