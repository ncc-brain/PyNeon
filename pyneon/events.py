from .tabular import BaseTabular
import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number
from typing import TYPE_CHECKING, Literal, Optional
import copy

if TYPE_CHECKING:
    from .stream import Stream


class Events(BaseTabular):
    """
    Base for event data (blinks, fixations, saccades, "events" messages).
    Timestamped by ``start timestamp [ns]`` or ``timestamp [ns]``.

    Parameters
    ----------
    data : pandas.DataFrame or pathlib.Path
        DataFrame or path to the CSV file containing the stream data.
        The data must be indexed by ``timestamp [ns]``.

    Attributes
    ----------
    file : pathlib.Path
        Path to the CSV file containing the event data.
    data : pandas.DataFrame
        DataFrame containing the event data.
    id_name : str
        Name of the column containing the event ID.
    """

    def __init__(self, data: pd.DataFrame | Path, id_name: str = None):
        if isinstance(data, Path):
            self.file = data
            data = pd.read_csv(data)
        else:
            self.file = None
        super().__init__(data)
        self.id_name = id_name

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
    ) -> Optional["Events"]:
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
        Events or None
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
            new_events = copy.copy(self)
            new_events.data = new_data
            return new_events

    def restrict(self, other: "Stream", inplace: bool = False) -> Optional["Events"]:
        """
        Restrict events to a time range defined by another stream.

        Parameters
        ----------
        other : Stream
            Stream to restrict to.

        Returns
        -------
        Events
            Restricted event data.
        """
        new_events = self.crop(
            other.first_ts, other.last_ts, by="timestamp", inplace=inplace
        )
        if new_events.data.empty:
            raise ValueError("No data found in the range of the other stream")
        return new_events

    def __getitem__(self, index) -> pd.Series:
        """Get an event series by index."""
        return self.data.iloc[index]
