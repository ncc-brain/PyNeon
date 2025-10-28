import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from warnings import warn
from typing import Callable
from typing_extensions import Literal

from .variables import native_to_cloud_column_map


def load_native_stream(
    recording_dir: str | Path,
    name: Literal["gaze", "imu", "eye_state", "eye_states"],
) -> pd.DataFrame:
    """
    Load native data from a recording directory.

    Parameters
    ----------
    recording_dir : str | Path
        _description_
    name : Literal["gaze", "imu", "eye_state"]
        Name of the stream to load.
        Must be one of "gaze", "imu", "eye_state", or "eye_states" (tolerated).

    Returns
    -------
    pd.DataFrame


    Raises
    ------
    FileNotFoundError
        _description_
    ValueError
        _description_
    """
    recording_dir = Path(recording_dir)
    name = "eye_state" if name == "eye_states" else name  # Tolerate plural form

    time_file = recording_dir / f"{name} ps1.time"
    raw_file = recording_dir / f"{name} ps1.raw"
    dtype_file = recording_dir / f"{name}.dtype"
    if name == "gaze":  # Use 200hz data if available
        time_200hz_file = recording_dir / "gaze_200hz.time"
        raw_200hz_file = recording_dir / "gaze_200hz.raw"
        if time_200hz_file.is_file() and raw_200hz_file.is_file():
            time_file = time_200hz_file
            raw_file = raw_200hz_file
    for file in [time_file, raw_file, dtype_file]:
        if not file.is_file():
            raise FileNotFoundError(
                f"Required {name} file {file.name} not found in {recording_dir}"
            )

    # Read timestamps
    ts = np.fromfile(time_file, dtype=np.int64)
    # Read data in the correct dtype
    dtype = np.dtype(literal_eval(dtype_file.read_text()))
    raw = np.fromfile(raw_file, dtype=dtype)
    if ts.shape[0] != raw.shape[0]:
        raise ValueError(
            f"Timestamp ({ts.shape[0]}) and data ({raw.shape[0]}) lengths do not match for {name} stream."
        )
    # Concat into a pandas dataframe
    data = pd.DataFrame(raw, index=ts)
    data.index.name = "timestamp [ns]"

    # Stream specific operations
    if name == "gaze":  # Try to attach `worn` column as in Cloud format
        try:
            worn_dtype = np.dtype(
                literal_eval((recording_dir / "worn.dtype").read_text())
            )
            worn = np.fromfile(
                raw_file.with_name(raw_file.name.replace("gaze", "worn")),
                dtype=worn_dtype,
            )["worn"]
            data["worn"] = worn.astype(np.int8)
        except Exception as e:
            warn(f"Could not load 'worn' data for gaze stream: {e}")
    elif name == "imu":  # Drop timestamp column duplicated in .time and .raw
        data.drop(columns=["timestamp_ns"], inplace=True, errors="ignore")
    # Rename columns to cloud format
    not_renamed_columns = (
        set(data.columns) - set(native_to_cloud_column_map.keys()) - {"worn"}
    )
    if not_renamed_columns:
        warn(
            "Following columns do not have a known alternative name in Pupil Cloud format. "
            "They will not be renamed: "
            f"{', '.join(not_renamed_columns)}"
        )
    data.rename(columns=native_to_cloud_column_map, errors="ignore", inplace=True)
    return data


def _check_data(data: pd.DataFrame) -> None:
    """
    Check if the data is in the correct format.
    """
    # Check if index name is timestamp [ns]
    if (data.index.name != "timestamp [ns]") and (
        data.index.name != "start timestamp [ns]"
    ):
        raise ValueError(
            "Index name must be 'timestamp [ns]' or 'start timestamp [ns]'"
        )

    # Check if index has duplicates
    if data.index.duplicated().any():
        data = data[~data.index.duplicated(keep="first")]
        print("Warning: Duplicated indices found and removed.")

    # Try to convert the index to int64
    try:
        data.index = data.index.astype("int64")
    except:
        raise ValueError(
            "Event index must be in UTC time in ns and thus convertible to int64"
        )

    # Sort by index
    data = data.sort_index(ascending=True)
    assert data.index.is_monotonic_increasing


def load_or_compute(
    path: Path,
    compute_fn: Callable[[], pd.DataFrame],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Load a DataFrame from a file or compute it if the file does not exist.
    """
    if path.is_file() and not overwrite:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            pd.read_json(path, orient="records", lines=True)
        elif path.suffix == ".pkl":
            df = pd.read_pickle(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if df.empty:
            raise ValueError(f"{path.name} is empty.")
        return df
    else:
        df = compute_fn()
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix == ".json":
            df.to_json(path, orient="records", lines=True)
        elif path.suffix == ".pkl":
            df.to_pickle(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if df.empty:
            raise ValueError(f"{path.name} is empty.")
        return df
