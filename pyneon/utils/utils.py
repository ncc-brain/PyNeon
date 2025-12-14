from pathlib import Path
from typing import Callable

import pandas as pd


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
    except Exception as e:
        raise ValueError(
            "Event index must be in UTC time in ns and thus convertible to int64. "
            f"Got error: {e}"
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
