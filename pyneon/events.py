from .tabular import NeonTabular
import numpy as np
import pandas as pd
from numbers import Number
from typing import TYPE_CHECKING, Literal, Optional
import copy

if TYPE_CHECKING:
    from .stream import NeonStream

from .utils import _check_event_data


class NeonEV(NeonTabular):
    """
    Base for Neon event data (blinks, fixations, saccades, "events" messages).
    """

    def __init__(self, file):
        super().__init__(file)

    @property
    def start_ts(self) -> np.ndarray:
        """Start timestamps of events in nanoseconds.."""
        return self.data.index.to_numpy()

    @property
    def end_ts(self) -> Optional[np.ndarray]:
        """End timestamps of events in nanoseconds."""
        if "end timestamp [ns]" in self.data.columns:
            return self.data["end timestamp [ns]"].to_numpy()
        else:
            raise ValueError("No 'end timestamp [ns]' column found in the instance.")

    @property
    def durations(self) -> Optional[np.ndarray]:
        """Duration of events in milliseconds."""
        if "duration [ms]" in self.data.columns:
            return self.data["duration [ms]"].to_numpy()
        else:
            raise ValueError("No 'duration [ms]' column found in the instance.")

    @property
    def id(self) -> Optional[np.ndarray]:
        """Event ID."""
        if self.id_name in self.data.columns and self.id_name is not None:
            return self.data[self.id_name].to_numpy()
        else:
            raise ValueError(f"No event ID column found in the instance.")

    def crop(
        self,
        tmin: Optional[Number] = None,
        tmax: Optional[Number] = None,
        by: Literal["timestamp", "row"] = "timestamp",
        inplace: bool = False,
    ) -> Optional["NeonEV"]:
        """
        Crop data to a specific time range based on timestamps or row numbers.

        Parameters
        ----------
        tmin : number, optional
            Start timestamp/row to crop the data to. If ``None``,
            the minimum timestamp/row in the data is used. Defaults to ``None``.
        tmax : number, optional
            End timestamp/row to crop the data to. If ``None``,
            the maximum timestamp/row in the data is used. Defaults to ``None``.
        by : "timestamp" or "row", optional
            Whether tmin and tmax are UTC timestamps in nanoseconds
            or row numbers of the stream data.
            Defaults to "timestamp".
        inplace : bool, optional
            Whether to replace the data in the object with the cropped data.
            Defaults to False.

        Returns
        -------
        NeonEV or None
            Cropped stream if ``inplace=False``, otherwise ``None``.
        """
        if tmin is None and tmax is None:
            raise ValueError("At least one of tmin or tmax must be provided")
        if by == "timestamp":
            t = self.start_ts
        else:
            t = np.arange(len(self))
        tmin = tmin if tmin is not None else t.min()
        tmax = tmax if tmax is not None else t.max()
        mask = (t >= tmin) & (t <= tmax)
        if not mask.any():
            raise ValueError("No data found in the specified time range")
        new_data = self.data[mask].copy()
        if inplace:
            self.data = new_data
        else:
            new_EV = copy.copy(self)
            new_EV.data = new_data
            return new_EV

    def restrict(
        self, other: "NeonStream", inplace: bool = False
    ) -> Optional["NeonEV"]:
        """
        Restrict events to a time range defined by another stream.

        Parameters
        ----------
        other : NeonStream
            Stream to restrict to.

        Returns
        -------
        NeonEV
            Restricted event data.
        """
        new_EV = self.crop(
            other.first_ts, other.last_ts, by="timestamp", inplace=inplace
        )
        if new_EV.data.empty:
            raise ValueError("No data found in the range of the other stream")
        return new_EV

    def __getitem__(self, index) -> pd.Series:
        """Get an event series by index."""
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


class CustomEvents(NeonEV):
    """
    Custom NeonEV class for user-defined event data.

    Parameters
    ----------
    data : pandas.DataFrame
        Event data. Must be indexed by 'timestamp [ns]' or 'start timestamp [ns]'.
    """

    file = None

    def __init__(self, data: pd.DataFrame):
        _check_event_data(data)
        self.data = data
        self.id_name = None
