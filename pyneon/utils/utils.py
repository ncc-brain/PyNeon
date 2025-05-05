import pandas as pd
import pathlib as Path
from typing import Callable


def _check_stream_data(data: pd.DataFrame) -> None:
    """
    Check if the data is in the correct format for a stream.
    """
    # Check if index name is timestamp [ns]
    if data.index.name != "timestamp [ns]":
        raise ValueError("Index name must be 'timestamp [ns]'")
    # Check if index is sorted
    if not data.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in increasing order")


def _check_event_data(data: pd.DataFrame) -> None:
    """
    Check if the data is in the correct format for an event.
    """
    # Check if index name is timestamp [ns]
    if (data.index.name != "timestamp [ns]") and (
        data.index.name != "start timestamp [ns]"
    ):
        raise ValueError(
            "Index name must be 'timestamp [ns]' or 'start timestamp [ns]'"
        )
    # Check if index is sorted
    if not data.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in increasing order")
    # Try to convert the index to int64
    try:
        data.index = data.index.astype("Int64")
    except:
        raise ValueError(
            "Event index must be in UTC time in ns and thus convertible to int64"
        )

def load_or_compute(
    path: Path,
    compute_fn: Callable[[], pd.DataFrame],
    overwrite: bool = False,
) -> pd.DataFrame:
    if path.is_file() and not overwrite:
        df = (
            pd.read_csv(path)
            if path.suffix == ".csv"
            else pd.read_json(path, orient="records", lines=True)
        )
        if df.empty:
            raise ValueError(f"{path.name} is empty.")
        return df
    else:
        df = compute_fn()
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_json(path, orient="records", lines=True)
        return df
