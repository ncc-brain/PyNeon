from pathlib import Path
from warnings import warn

import pandas as pd

from .utils.variables import data_types


class BaseTabular:
    """
    Base for tabular data classes like :class:`Events` and :class:`Stream`.

    When initialized, performs the following checks and processing on the input data:

    1. Ensures the data is not empty.
    2. If present, checks that ``recording id`` and ``section id`` columns contain
       only a single unique value, and removes these columns.
    3. Identifies and handles duplicate columns by retaining only the first occurrence
       and issuing a warning.
    4. Sets the data types of columns based on a predefined mapping. Columns not found
       in this mapping are left with their default data types, and a warning is issued.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data to be processed.

    Attributes
    ----------
    data : pandas.DataFrame
        The processed data.
    """

    def __init__(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("Data is empty.")

        if "recording id" in data.columns:
            if data["recording id"].nunique() > 1:
                raise ValueError("Data contains multiple recording IDs")
            data = data.drop(columns=["recording id"])

        if "section id" in data.columns:
            if data["section id"].nunique() > 1:
                raise ValueError("Data contains multiple section IDs")
            data = data.drop(columns=["section id"])

        # Deal with duplicate columns
        duplicate_cols = data.columns[data.columns.duplicated()].unique()
        if len(duplicate_cols) > 0:
            warn(
                "Data contains duplicate columns: "
                f"{', '.join(duplicate_cols)}. "
                "Using the first occurrence.",
                UserWarning,
            )
            data = data.loc[:, ~data.columns.duplicated()]

        # Set data types
        unknown_cols = []
        for col in data.columns:
            if col not in data_types:
                unknown_cols.append(col)
            else:
                data[col] = data[col].astype(data_types[col])
        if unknown_cols:
            warn(
                "Following columns not in known data types, using default data types: "
                f"{', '.join(unknown_cols)}"
            )

        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Return a tuple representing the shape of the data.

        Returns
        -------
        tuple[int, int]
            Shape of the data as (rows, columns).
        """
        return self.data.shape

    @property
    def columns(self) -> pd.Index:
        """Return the column labels of the data.

        Returns
        -------
        pd.Index
            Column labels of the data.
        """
        return self.data.columns

    @property
    def dtypes(self) -> pd.Series:
        """Return the data types of the columns in the data.

        Returns
        -------
        pd.Series
            Data types of the columns in the data.
        """
        return self.data.dtypes

    def copy(self):
        """Create a deep copy of the instance."""
        from copy import deepcopy

        return deepcopy(self)

    def save(self, output_path: str | Path):
        """Save the data to a CSV file.

        Data types and index are preserved on reload (round-trip safe).

        Parameters
        ----------
        output_path : str or pathlib.Path
            Path to the output CSV file.

        Examples
        --------
        >>> gaze = recording.gaze
        >>> gaze.save("gaze_data.csv")
        >>> gaze_reloaded = Stream("gaze_data.csv")
        """
        # Use a high-precision float format to make CSV round-trips stable.
        self.data.to_csv(output_path, index=True, header=True, float_format="%.18g")
