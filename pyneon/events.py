from .tabular import NeonTabular
import pandas as pd


class NeonEV(NeonTabular):
    """
    Base for Neon event data (blinks, fixations, saccades, "events" messages).
    """

    def __init__(self, file):
        super().__init__(file)

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
