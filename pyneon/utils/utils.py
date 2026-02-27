import numpy as np
import pandas as pd
from typing import Iterable, Optional


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


def _validate_neon_tabular_data(data: pd.DataFrame) -> None:
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


def _validate_df_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    include_index_name: bool = False,
    df_name: Optional[str] = None,
) -> None:
    """Verify that the DataFrame contains all expected columns (including index)."""
    for required_col in required_columns:
        df_cols = set(df.columns)
        if include_index_name and df.index.name:
            df_cols.add(df.index.name)
        if required_col not in df_cols:
            df_name_str = f" '{df_name}'" if df_name else ""
            raise ValueError(
                f"DataFrame{df_name_str} must contain '{required_col}' column."
            )
