from ast import literal_eval
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd

from .tabular import BaseTabular
from .utils.doc_decorators import inplace_doc
from .utils.variables import native_to_cloud_column_map

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
        data = pd.DataFrame(
            {"timestamp [ns]": ts, "name": event_names, "type": "recording"}
        )
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
            idx_name = f"{type[:-1]} id"
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
        elif "blink" in file_path.stem:
            idx_name = "blink id"
        else:
            raise ValueError(f"Unknown native events file: {file_path.name}")
        # Reset index to count events
        data = pd.DataFrame(raw)

        # Drop "event_type" column if exists
        if "event_type" in data.columns:
            data.drop(columns=["event_type"], inplace=True)

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

        # Add event ID column
        data[idx_name] = np.arange(len(data)) + 1

        # Add duration column
        durations = (
            data["end timestamp [ns]"] - data["start timestamp [ns]"]
        ) // 1_000_000
        data["duration [ms]"] = durations.astype("int64")

    return data, files


def _infer_events_type_and_id(data: pd.DataFrame) -> tuple[str, Optional[str]]:
    """
    Infer event type based on presence of specific columns.
    If multiple or no matches found, return "custom".
    """
    col_map = {
        "blink id": "blinks",
        "saccade id": "saccades",
        "fixation id": "fixations",
        "name": "events",
    }
    reverse_map = {v: k for k, v in col_map.items()}

    # Find all matching event types in the dataframe
    types = {col_map[c] for c in data.columns if c in col_map}
    if len(types) != 1:
        # None or more than one match → custom event type
        return "custom", None
    type = types.pop()
    id_name = None if type == "events" else reverse_map[type]
    return type, id_name


class Events(BaseTabular):
    """
    Container for discrete event data (e.g., blinks, fixations, saccades, or
    message events).

    Parameters
    ----------
    source : pandas.DataFrame or pathlib.Path or str
        Source of the event data. Can be either:

        * :class:`pandas.DataFrame`: Must contain appropriate event columns.
        * :class:`pathlib.Path` or :class:`str`: Path to an event data file.
          Supported file formats:

        - ``.csv``: Pupil Cloud format file.
        - ``.raw`` / ``.txt``: Native Pupil Labs format (requires
          corresponding ``.time`` and ``.dtype`` files in the same directory).

        Note: Native format columns are automatically renamed to Pupil Cloud
        format for consistency. For example, ``mean_gaze_x`` -> ``fixation x [px]``.

    type : {"fixations", "saccades"}, optional
        Required only when loading from native ``fixations ps1.raw`` file, to
        explicitly specify whether to interpret the data as fixation or
        saccade events. Ignored for other formats.

    Attributes
    ----------
    file : pathlib.Path or None
        Path to the source file(s). ``None`` if initialized from a DataFrame.
    data : pandas.DataFrame
        Event data with standardized column names.
    type : {"blinks", "fixations", "saccades", "events", "custom"}
        Inferred event type based on data columns.
    id_name : str or None
        Column name holding event IDs (e.g., ``blink id``, ``fixation id``,
        ``saccade id``). ``None`` for ``events`` and ``custom`` types.

    Examples
    --------
    Load from Pupil Cloud CSV:

    >>> blinks = Events("blinks.csv")

    Load from native format:

    >>> blinks = Events("blinks ps1.raw")
    >>> fixations = Events("fixations ps1.raw", type="fixations")
    >>> events = Events("event.txt")

    Create from DataFrame:

    >>> df = pd.DataFrame({"start timestamp [ns]": [...], "end timestamp [ns]": [...]})
    >>> saccades = Events(df)
    """

    def __init__(self, source: pd.DataFrame | Path | str, type: Optional[str] = None):
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            if not source.is_file():
                raise FileNotFoundError(f"File does not exist: {source}")
            if source.suffix == ".csv":
                self.file = source
                data = pd.read_csv(source)
            elif source.suffix in [".txt", ".raw"]:
                data, self.file = _load_native_events_data(source, type)
        else:  # pd.DataFrame
            data = source.copy(deep=True)
            self.file = None
        super().__init__(data)
        self.type, self.id_name = _infer_events_type_and_id(self.data)

    def __getitem__(self, index) -> pd.Series:
        """Get an event series by index."""
        return self.data.iloc[index]

    @property
    def start_ts(self) -> np.ndarray:
        """
        Start timestamps of events in nanoseconds.

        Raises
        ------
        ValueError
            If no ``start timestamp [ns]`` or ``timestamp [ns]`` column is found in the instance.
        """
        if self.type == "events":
            return self.data["timestamp [ns]"].to_numpy()
        if "start timestamp [ns]" in self.data.columns:
            return self.data["start timestamp [ns]"].to_numpy()
        else:
            raise ValueError("No `start timestamp [ns]` column found in the instance.")

    @property
    def end_ts(self) -> np.ndarray:
        """
        End timestamps of events in nanoseconds.

        Raises
        ------
        ValueError
            If no ``end timestamp [ns]`` column is found in the instance.
        """
        if "end timestamp [ns]" in self.data.columns:
            return self.data["end timestamp [ns]"].to_numpy()
        else:
            raise ValueError("No `end timestamp [ns]` column found in the instance.")

    @property
    def durations(self) -> np.ndarray:
        """
        Duration of events in milliseconds.

        Raises
        ------
        ValueError
            If no ``duration [ms]`` column is found in the instance.
        """
        if "duration [ms]" in self.data.columns:
            return self.data["duration [ms]"].to_numpy()
        else:
            raise ValueError("No `duration [ms]` column found in the instance.")

    @property
    def id(self) -> np.ndarray:
        """
        Event ID.

        Raises
        ------
        ValueError
            If no ID column (e.g., ``<xxx> id``) is found in the instance.
        """
        if self.id_name in self.data.columns and self.id_name is not None:
            return self.data[self.id_name].to_numpy()
        else:
            raise ValueError("No ID column (e.g., `<xxx> id`) found in the instance.")

    @inplace_doc
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
        {inplace_doc}

        Returns
        -------
        Events or None
            Cropped events if ``inplace=False``, otherwise ``None``.
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

        inst = self if inplace else self.copy()
        inst.data = self.data[mask].copy()
        return None if inplace else inst

    @inplace_doc
    def restrict(self, other: "Stream", inplace: bool = False) -> Optional["Events"]:
        """
        Temporally crop the events to the range of timestamps a stream.
        Equivalent to ``crop(other.first_ts, other.last_ts)``.

        Parameters
        ----------
        other : Stream
            Stream to restrict to.
        {inplace_doc}

        Returns
        -------
        Events or None
            Restricted events if ``inplace=False``, otherwise ``None``.
        """
        return self.crop(other.first_ts, other.last_ts, by="timestamp", inplace=inplace)

    @inplace_doc
    def filter_by_duration(
        self,
        dur_min: Optional[Number] = None,
        dur_max: Optional[Number] = None,
        reset_id: bool = True,
        inplace: bool = False,
    ) -> Optional["Events"]:
        """
        Filter events by their durations. Useful for removing very short or long events.

        Parameters
        ----------
        dur_min : number, optional
            Minimum duration (in milliseconds) of events to keep.
            If ``None``, no minimum duration filter is applied. Defaults to ``None``.
        dur_max : number, optional
            Maximum duration (in milliseconds) of events to keep.
            If ``None``, no maximum duration filter is applied. Defaults to ``None``.
        reset_id : bool, optional
            Whether to reset event IDs after filtering. Also resets the DataFrame index.
            Defaults to ``True``.
        {inplace_doc}

        Returns
        -------
        Events or None
            Filtered events if ``inplace=False``, otherwise ``None``.
        """
        if dur_min is None and dur_max is None:
            raise ValueError("At least one of dur_min or dur_max must be provided")
        dur_min = dur_min if dur_min is not None else self.durations.min()
        dur_max = dur_max if dur_max is not None else self.durations.max()
        mask = (self.durations >= dur_min) & (self.durations <= dur_max)
        if not mask.any():
            raise ValueError("No data found in the specified duration range")
        print(f"Filtering out {len(self) - mask.sum()} out of {len(self)} events.")
        inst = self if inplace else self.copy()
        inst.data = self.data[mask].copy()
        if reset_id:
            if self.id_name is not None:
                inst.data[self.id_name] = np.arange(len(inst.data)) + 1
                inst.data.reset_index(drop=True, inplace=True)
            else:
                raise KeyError(
                    "Cannot reset event IDs as no event ID column is known for this instance."
                )
        return None if inplace else inst
