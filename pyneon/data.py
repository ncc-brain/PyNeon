from pathlib import Path
import pandas as pd


class NeonTabular:
    """
    Base for Neon tabular data. It reads from a CSV file and stores the data
    as a pandas DataFrame (with section and recording IDs removed). The `timestamp [ns]`
    (for streams) or `start timestamp [ns]` (for events) column is set as the index.
    """

    def __init__(self, file: Path):
        self.file = file
        if isinstance(file, Path) and file.suffix == ".csv":
            data = pd.read_csv(file)
        else:  # TODO: Implement reading native data formats
            pass

        if data["recording id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple recording IDs")
        self.data = data.drop(columns=["recording id"])

        # Every data file except events.csv has a section id column
        if "section id" in self.data.columns:
            if data["section id"].nunique() > 1:
                raise ValueError(f"{file.name} contains multiple section IDs")
            self.data.drop(columns=["section id"], inplace=True)

        # Set the timestamp column as the index
        if "timestamp [ns]" in self.data.columns:
            self.data.set_index("timestamp [ns]", inplace=True)
        elif "start timestamp [ns]" in self.data.columns:
            self.data.set_index("start timestamp [ns]", inplace=True)
        else:
            raise ValueError(f"{file.name} does not contain a timestamp column")
        assert pd.api.types.is_integer_dtype(self.data.index.dtype)
        self.data.sort_index(inplace=True)

    def __len__(self) -> int:
        return self.data.shape[0]
