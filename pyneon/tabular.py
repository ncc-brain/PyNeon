from pathlib import Path
import pandas as pd


class NeonTabular:
    """
    Base for Neon tabular data. It reads from a CSV file and stores the data
    as a pandas DataFrame (with section and recording IDs removed). The `timestamp [ns]`
    (for streams) or `start timestamp [ns]` (for events) column is set as the index.

    Parameters
    ----------
    file : Path
        Path to the CSV file.

    Attributes
    ----------
    data : pandas.DataFrame
        The processed data with the timestamp index.
    """

    def __init__(self, file: Path):
        self.file = file
        if isinstance(file, Path) and file.suffix == ".csv":
            data = pd.read_csv(file)
        else:  # TODO: Implement reading native data formats
            raise NotImplementedError("Reading non-CSV files is not yet implemented.")

        if data.empty:
            raise ValueError(f"The data file '{file.name}' is empty.")

        if data["recording id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple recording IDs")
        data = data.drop(columns=["recording id"])

        # Every data file except events.csv has a section id column
        if "section id" in data.columns:
            if data["section id"].nunique() > 1:
                raise ValueError(f"{file.name} contains multiple section IDs")
            data = data.drop(columns=["section id"])

        # Set the timestamp column as the index
        if "timestamp [ns]" in data.columns:
            data = data.set_index("timestamp [ns]")
        elif "start timestamp [ns]" in data.columns:
            data = data.set_index("start timestamp [ns]")
        else:
            raise ValueError(f"{file.name} does not contain a timestamp column")

        # Ensure the index is of integer type and sorted
        if not pd.api.types.is_integer_dtype(data.index.dtype):
            raise TypeError("The index must be of integer type.")

        data = data.sort_index()
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]
