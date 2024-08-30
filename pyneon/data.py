from pathlib import Path
import pandas as pd
import numpy as np


class NeonData:
    """Holder for all formats Neon tabular data."""

    def __init__(self, file: Path):
        self.file = file
        if isinstance(file, Path) and file.suffix == ".csv":
            data = pd.read_csv(file)
        else:  # TODO: Implement reading native data formats
            pass
        # Assert that all section id are the same
        if data["section id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple section IDs")
        if data["recording id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple recording IDs")
        self.data = data.drop(columns=["section id", "recording id"])

    def __len__(self) -> int:
        return self.data.shape[0]


class NeonSignal(NeonData):
    """Holder for continuous signals."""

    def __init__(self, file):
        super().__init__(file)

    def timestamps(self) -> np.ndarray:
        """Returns the timestamps of the stream in nanoseconds."""
        return self.data["timestamp [ns]"].to_numpy()

    def duration(self) -> float:
        """Returns the duration of the stream in seconds."""
        return (self.timestamps()[-1] - self.timestamps()[0]) / 1e9

    def sampling_freq_effective(self) -> float:
        """Returns the effective sampling frequency of the stream in Hz."""
        return self.data.shape[0] / self.duration()


class NeonGaze(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 200


class NeonIMU(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 120


class NeonEyeStates(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 200


class NeonEvents(NeonData):
    def __init__(self, file):
        super().__init__(file)


class NeonBlinks(NeonEvents):
    def __init__(self, file):
        super().__init__(file)


class NeonFixations(NeonData):
    def __init__(self, file):
        super().__init__(file)


class NeonLabels(NeonData):
    def __init__(self, file):
        super().__init__(file)
