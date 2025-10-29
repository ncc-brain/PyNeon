import pandas as pd
from warnings import warn

from .utils import data_types


class BaseTabular:
    """
    Base for Neon tabular data. It reads from a CSV file and stores the data
    as a pandas DataFrame (with section and recording IDs removed). The `timestamp [ns]`
    (for streams) or `start timestamp [ns]` (for events) column is set as the index.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data to be processed.

    Attributes
    ----------
    data : pandas.DataFrame
        The processed data with the timestamp index.
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

        # Set the timestamp column as the index if not already
        valid_index_names = {"timestamp [ns]", "start timestamp [ns]"}
        if data.index.name not in valid_index_names:
            valid = False
            for col in valid_index_names:
                if col in data.columns:
                    data = data.set_index(col)
                    valid = True
                    break
            if not valid:
                raise ValueError("Data does not contain a valid timestamp column")

        # Ensure the index is of integer type and sorted
        if not pd.api.types.is_integer_dtype(data.index.dtype):
            raise ValueError(
                "Data index must be in UTC time in ns and thus convertible to int64"
            )
        else:
            data.index = data.index.astype("int64")

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

        data = data.sort_index()
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
