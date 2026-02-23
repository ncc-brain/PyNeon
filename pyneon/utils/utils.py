from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform 2D points by a 3x3 homography.

    Parameters
    ----------
    points : numpy.ndarray of shape (N, 2)
        2D points to be transformed.
    H : numpy.ndarray of shape (3, 3)
        Homography matrix.

    Returns
    -------
    numpy.ndarray of shape (N, 2)
        Transformed 2D points.
    """
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed_h = (H @ points_h.T).T
    transformed_2d = transformed_h[:, :2] / transformed_h[:, 2:]
    return transformed_2d


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
            "Event index must be in Unix time in ns and thus convertible to int64. "
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
            df = pd.read_json(path, orient="records", lines=True)
        elif path.suffix == ".pkl":
            df = pd.read_pickle(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if df.empty:
            raise ValueError(f"{path.name} is empty.")
        return df
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
