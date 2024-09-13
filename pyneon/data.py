from pathlib import Path
import pandas as pd


class NeonData:
    """
    Base for Neon tabular data. It reads from a CSV file and stores the data
    as a pandas DataFrame (with section and recording IDs removed).
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

    def __len__(self) -> int:
        return self.data.shape[0]
