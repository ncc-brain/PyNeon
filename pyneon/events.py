from .tabular import NeonTabular
import numpy as np
import pandas as pd


class NeonEV(NeonTabular):
    """
    Base for Neon event data (blinks, fixations, saccades, "events" messages).
    """

    def __init__(self, file):
        super().__init__(file)

    @property
    def start_ts(self) -> np.ndarray:
        """Start timestamps of events in nanoseconds.."""
        return self.index.to_numpy()

    @property
    def end_ts(self) -> np.ndarray:
        """End timestamps of events in nanoseconds."""
        if "end timestamp [ns]" in self.data.columns:
            return self.data["end timestamp [ns]"].to_numpy()
        else:
            print("No 'end timestamp [ns]' column found.")
            return np.empty(self.data.shape[0], dtype=np.int64)

    @property
    def durations(self) -> np.ndarray:
        """Duration of events in milliseconds."""
        if "duration [ms]" in self.data.columns:
            return self.data["duration [ms]"].to_numpy()
        else:
            print("No 'duration [ms]' column found.")
            return np.empty(self.data.shape[0], dtype=np.int64)
        
    @property
    def id(self) -> np.ndarray:
        """Event ID."""
        if self.id_name in self.data.columns and self.id_name is not None:
            return self.data[self.id_name].to_numpy()
        else:
            print(f"Event ID name is undefined or not found in the data.")
            return np.empty(self.data.shape[0], dtype=np.int64)

    def __getitem__(self, index) -> pd.Series:
        """Get an event timeseries by index."""
        return self.data.iloc[index]


class NeonBlinks(NeonEV):
    """Blink data."""

    def __init__(self, file):
        super().__init__(file)
        if self.data.index.name != "start timestamp [ns]":
            raise ValueError("Blink data should be indexed by `start timestamp [ns]`.")
        self.data = self.data.astype(
            {
                "blink id": "Int32",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
            }
        )
        self.id_name = "blink id"


class NeonFixations(NeonEV):
    """Fixation data."""

    def __init__(self, file):
        super().__init__(file)
        if self.data.index.name != "start timestamp [ns]":
            raise ValueError(
                "Fixation data should be indexed by `start timestamp [ns]`."
            )
        self.data = self.data.astype(
            {
                "fixation id": "Int32",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
                "fixation x [px]": float,
                "fixation y [px]": float,
                "azimuth [deg]": float,
                "elevation [deg]": float,
            }
        )
        self.id_name = "fixation id"


class NeonSaccades(NeonEV):
    """Saccade data."""

    def __init__(self, file):
        super().__init__(file)
        if self.data.index.name != "start timestamp [ns]":
            raise ValueError(
                "Saccade data should be indexed by `start timestamp [ns]`."
            )
        self.data = self.data.astype(
            {
                "saccade id": "Int32",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
                "amplitude [px]": float,
                "amplitude [deg]": float,
                "mean velocity [px/s]": float,
                "peak velocity [px/s]": float,
            }
        )
        self.id_name = "saccade id"


class NeonEvents(NeonEV):
    """Event data."""

    def __init__(self, file):
        super().__init__(file)
        if self.data.index.name != "timestamp [ns]":
            raise ValueError("Event data should be indexed by `timestamp [ns]`.")
        self.data = self.data.astype(
            {
                "name": str,
                "type": str,
            }
        )
        self.id_name = None
