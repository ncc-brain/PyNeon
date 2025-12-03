import pandas as pd
from warnings import warn

from .utils import data_types


class BaseTabular:
    """
    Base for Neon tabular data. It takes a Pandas DataFrame, strips unnecessary
    columns (section and recording IDs), and sets the correct data types for known columns.

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
        return self.data.shape

    @property
    def columns(self) -> pd.Index:
        """Column names of the stream data."""
        return self.data.columns

    @property
    def dtypes(self) -> pd.Series:
        """Data types of the columns in the stream data."""
        return self.data.dtypes

    def copy(self):
        """Create a deep copy of the instance."""
        from copy import deepcopy

        return deepcopy(self)
