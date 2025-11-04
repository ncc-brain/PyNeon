from .tabular import BaseTabular
import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number
from typing import TYPE_CHECKING, Literal, Optional
from ast import literal_eval
from warnings import warn
import copy

from .utils import native_to_cloud_column_map

if TYPE_CHECKING:
    from .stream import Stream


def _load_native_events_data(
    file_path: Path, type: Optional[str] = None
) -> tuple[pd.DataFrame, list[Path]]:
    rec_dir = file_path.parent.resolve()

    # Read data in the correct dtype
    if file_path.stem == "event":
        # Only events need explicit .time file for timestamps
        time_file = file_path.with_suffix(".time")
        if not time_file.is_file():
            raise FileNotFoundError(f"Missing .time file {time_file.name} in {rec_dir}")
        ts = np.fromfile(time_file, dtype=np.int64)

        # Read event names
        with open(file_path, "r") as f:
            event_names = [line.strip() for line in f.readlines()]
        data = pd.DataFrame({"name": event_names}, index=ts)
        data["type"] = "recording"
        data.index.name = "timestamp [ns]"
        files = [file_path, time_file]
    else:
        dtype_file = next(rec_dir.glob(f"{file_path.name[:3]}*.dtype"), None)
        if dtype_file is None:
            raise FileNotFoundError(
                f"Missing .dtype file for {file_path.name} in {rec_dir}"
            )
        files = [file_path, dtype_file]

        dtype = np.dtype(literal_eval(dtype_file.read_text()))
        raw = np.fromfile(file_path, dtype=dtype)
        if "fixation" in file_path.stem:
            if type is None or type not in ["saccades", "fixations"]:
                raise ValueError("Type ('saccades'/'fixations') must be specified")
            if type == "fixations":
                mask = raw["event_type"] == 1
                raw = raw[mask]
                raw = raw[
                    [
                        col
                        for col in raw.dtype.names
                        if "mean_gaze" in col or "timestamp" in col
                    ]
                ]
            else:  # type == "saccades"
                mask = raw["event_type"] == 0
                raw = raw[mask]
                raw = raw[
                    [
                        col
                        for col in raw.dtype.names
                        if col
                        in [
                            "start_timestamp_ns",
                            "end_timestamp_ns",
                            "amplitude_pixels",
                            "amplitude_angle_deg",
                            "mean_velocity",
                            "max_velocity",
                        ]
                    ]
                ]
        # Get all column names from raw
        data = (
            pd.DataFrame(raw)
            .rename(columns={"start_timestamp_ns": "start timestamp [ns]"})
            .set_index("start timestamp [ns]")
        )

        # Rename columns to cloud format
        not_renamed_columns = set(data.columns) - set(native_to_cloud_column_map.keys())
        if not_renamed_columns:
            warn(
                "Following columns do not have a known alternative name in Pupil Cloud format. "
                "They will not be renamed: "
                f"{', '.join(not_renamed_columns)}",
                UserWarning,
            )
        data.rename(columns=native_to_cloud_column_map, errors="ignore", inplace=True)

    return data, files


def _infer_events_name(data: pd.DataFrame) -> str:
    """
    Infer events name based on presence of specific columns.
    If multiple or no matches found, return "custom".
    """
    col_map = {
        "blink id": "blinks",
        "saccade id": "saccades",
        "fixation id": "fixations",
        "name": "events",
    }
    names = {col_map[c] for c in data.columns if c in col_map}
    return names.pop() if len(names) == 1 else "custom"


class Events(BaseTabular):
    """
    Base for event data (blinks, fixations, saccades, "events" messages).
    Timestamped by ``start timestamp [ns]`` or ``timestamp [ns]``.

    Parameters
    ----------
    data : pandas.DataFrame or pathlib.Path
        DataFrame or path to the CSV file containing the stream data.
        The data must be indexed by ``timestamp [ns]``.
    name : str, optional
        Name of the event type. Defaults to "custom".
    id_name : str, optional
        Name of the column containing the event ID. Defaults to None.
        If None, the event ID is not included in the data.

    Attributes
    ----------
    file : pathlib.Path
        Path to the CSV file containing the event data.
    data : pandas.DataFrame
        DataFrame containing the event data.
    name : str
        Name of the event type.
    id_name : str
        Name of the column containing the event ID.
    """

    def __init__(self, data: pd.DataFrame | Path, type: Optional[str] = None):
        if isinstance(data, Path):
            if not data.is_file():
                raise FileNotFoundError(f"File not exist: {data}")
            if data.suffix == ".csv":
                self.file = data
                data = pd.read_csv(data)
            elif data.suffix in [".txt", ".raw"]:
                data, self.file = _load_native_events_data(data, type)
        else:
            self.file = None
        super().__init__(data)
        self.name = _infer_events_name(self.data)

    def __getitem__(self, index) -> pd.Series:
        """Get an event series by index."""
        return self.data.iloc[index]

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
            raise ValueError("No event ID column found in the instance.")

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
        tmin = t.min() if tmin is None else tmin
        tmax = t.max() if tmax is None else tmax
        mask = (t >= tmin) & (t <= tmax)
        if not mask.any():
            raise ValueError("No data found in the specified time range")
        if inplace:
            self.data = self.data[mask].copy()
        else:
            new_events = copy.copy(self)
            new_events.data = self.data[mask].copy()
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
        return new_events
