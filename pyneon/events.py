from ast import literal_eval
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd

from .tabular import BaseTabular
from .utils import _apply_homography
from .utils.doc_decorators import fill_doc
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


def _infer_events_type(data: pd.DataFrame) -> str:
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
        data.index = pd.RangeIndex(start=0, stop=len(data), name="event id")
        return "custom"
    type = types.pop()
    if type == "events":
        data.index.name = "event id"
    else:
        data.set_index(reverse_map[type], inplace=True)
    return type


class Events(BaseTabular):
    """
    Container for discrete event data (e.g., blinks, fixations, saccades, or
    message events).

    Parameters
    ----------
    source : pandas.DataFrame or pathlib.Path or str
        Source of the event data. Can be either:

        - :class:`pandas.DataFrame`: Direct input of event data.
        - :class:`pathlib.Path` or :class:`str`: Path to an event data file.
          Supported file formats:

          - ``.csv``: Pupil Cloud format CSV file.
          - ``.raw`` / ``.txt``: Native format (requires corresponding
            ``.time`` and ``.dtype`` files in the same directory).

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
    type : str
        Inferred event type based on data columns.

    Examples
    --------
    Load from Pupil Cloud CSV:

    >>> blinks = Events("blinks.csv")

    Load from native format:

    >>> blinks = Events("blinks ps1.raw")
    >>> fixations = Events("fixations ps1.raw", type="fixations")
    >>> saccades = Events("fixations ps1.raw", type="saccades")
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
        self.type = _infer_events_type(self.data)

    def __getitem__(self, index) -> pd.Series:
        """Get an event series by index."""
        return self.data.iloc[index]

    def __repr__(self) -> str:
        return f"""Events type: {self.type}
Number of samples: {len(self)}
Columns: {list(self.data.columns)}
"""

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
            return self.data["timestamp [ns]"].to_numpy(np.int64)
        if "start timestamp [ns]" in self.data.columns:
            return self.data["start timestamp [ns]"].to_numpy(np.int64)
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
            return self.data["end timestamp [ns]"].to_numpy(np.int64)
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
        Event IDs.
        """
        return self.data.index.to_numpy(np.int32)

    @property
    def id_name(self) -> Optional[str]:
        """
        Name of the event ID column based on event type.

        Returns
        -------
        str or None
            The ID column name (e.g., ``"blink id"``, ``"fixation id"``, ``"saccade id"``,
            ``"event id"``) for known event types, or ``None`` for custom event types.
        """
        id_map = {
            "blinks": "blink id",
            "fixations": "fixation id",
            "saccades": "saccade id",
            "events": "event id",
        }
        return id_map.get(self.type, None)

    @fill_doc
    def crop(
        self,
        tmin: Optional[Number] = None,
        tmax: Optional[Number] = None,
        by: Literal["timestamp", "sample"] = "timestamp",
        inplace: bool = False,
    ) -> Optional["Events"]:
        """
        Extract a subset of events within a specified temporal range.

        The ``by`` parameter determines how ``tmin`` and ``tmax`` are interpreted:
        - ``"timestamp"``: Absolute Unix timestamps in nanoseconds (based on event start times)
        - ``"sample"``: Zero-based event indices

        Both bounds are inclusive. If either bound is omitted, it defaults to the
        events' natural boundary (earliest or latest event).

        Parameters
        ----------
        tmin : numbers.Number, optional
            Lower bound of the range to extract (inclusive). If ``None``,
            starts from the first event. Defaults to ``None``.
        tmax : numbers.Number, optional
            Upper bound of the range to extract (inclusive). If ``None``,
            extends to the last event. Defaults to ``None``.
        by : {"timestamp", "sample"}, optional
            Unit used to interpret ``tmin`` and ``tmax``. Defaults to ``"timestamp"``.

        %(inplace)s

        Returns
        -------
        %(events_or_none)s

        Raises
        ------
        ValueError
            If both ``tmin`` and ``tmax`` are ``None``, or if no events
            fall within the specified range.

        Examples
        --------
        Crop fixations to the first 5 seconds:

        >>> fixations_5s = fixations.crop(tmin=rec.gaze.first_ts, 
        ...                                tmax=rec.gaze.first_ts + 5e9, 
        ...                                by="timestamp")

        Extract the first 100 blinks:

        >>> first_100 = blinks.crop(tmin=0, tmax=99, by="sample")
        """
        if tmin is None and tmax is None:
            raise ValueError("At least one of `tmin` or `tmax` must be provided")
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

    @fill_doc
    def restrict(self, other: "Stream", inplace: bool = False) -> Optional["Events"]:
        """
        Align events to match a stream's temporal range.

        This method filters events to include only those whose start times fall
        between the first and last timestamps of the reference stream. It is
        equivalent to calling
        ``crop(tmin=other.first_ts, tmax=other.last_ts, by="timestamp")``.

        Useful for limiting event analysis to periods when a particular data stream
        is available.

        Parameters
        ----------
        other : Stream
            Reference stream whose temporal boundaries define the cropping range.

        %(inplace)s

        Returns
        -------
        %(events_or_none)s

        Examples
        --------
        Analyze only blinks that occurred during recorded gaze data:

        >>> blinks_with_gaze = blinks.restrict(gaze)
        """
        return self.crop(other.first_ts, other.last_ts, by="timestamp", inplace=inplace)

    @fill_doc
    def filter_by_duration(
        self,
        dur_min: Optional[Number] = None,
        dur_max: Optional[Number] = None,
        reset_id: bool = False,
        inplace: bool = False,
    ) -> Optional["Events"]:
        """
        Filter events by their durations. Useful for removing very short or long events.

        Parameters
        ----------
        dur_min : number, optional
            Minimum duration (in milliseconds) of events to keep (inclusive).
            If ``None``, no minimum duration filter is applied. Defaults to ``None``.
        dur_max : number, optional
            Maximum duration (in milliseconds) of events to keep (inclusive).
            If ``None``, no maximum duration filter is applied. Defaults to ``None``.
        reset_id : bool, optional
            Whether to reset event IDs after filtering.
            Defaults to ``False``.

        %(inplace)s

        Returns
        -------
        %(events_or_none)s
        """
        if "duration [ms]" not in self.data.columns:
            raise ValueError("No `duration [ms]` column found in the instance.")
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
            # Reset without losing original index name
            inst.data.index = pd.RangeIndex(
                start=0, stop=len(inst.data), name=inst.data.index.name
            )
        return None if inplace else inst

    @fill_doc
    def filter_by_name(
        self,
        names: str | list[str],
        col_name: str = "name",
        reset_id: bool = False,
        inplace: bool = False,
    ) -> Optional["Events"]:
        """
        Filter events by matching values in a specified column.
        Designed primarily for filtering :attr:`Recording.events` by their names.

        This method selects only the events whose value in ``col_name`` matches
        one or more of the provided ``names``. If no events match, a
        ``ValueError`` is raised.

        Parameters
        ----------
        names : str or list of str
            Event name or list of event names to keep. Matching is exact
            and case-sensitive.
        col_name : str, optional
            Name of the column in ``self.data`` to use for filtering.
            Must exist in the ``Events`` instance's DataFrame.
            Defaults to ``"name"``.
        reset_id: bool = False, optional
            Whether to reset event IDs after filtering.
            Defaults to ``False``.
        %(inplace)s

        Returns
        -------
        %(events_or_none)s
        """
        if col_name not in self.data.columns:
            raise KeyError(f"No `{col_name}` column found in the instance.")

        names = [names] if isinstance(names, str) else names
        mask = self.data[col_name].isin(names)
        if not mask.any():
            raise ValueError(
                f"No data found matching the specified event names {names}"
            )

        inst = self if inplace else self.copy()
        inst.data = self.data[mask].copy()
        if reset_id:
            inst.data.index = pd.RangeIndex(
                start=0, stop=len(inst.data), name=inst.data.index.name
            )
        return None if inplace else inst

    @fill_doc
    def apply_homographies(
        self,
        homographies: "Stream",
        max_gap_ms: Number = 500,
        overwrite: bool = False,
        inplace: bool = False,
    ) -> Optional["Events"]:
        """
        Compute fixation locations in surface coordinates using provided homographies
        based on fixation pixel coordinates and append them to the events data.

        Since homographies are estimated per video frame and might not be available
        for every frame, they need to be resampled/interpolated to the timestamps of the
        fixation data before application.

        The events data must contain the required fixation columns:
        ``fixation x [px]`` and ``fixation y [px]``.

        Parameters
        ----------
        homographies : Stream
            Stream containing homography matrices with columns ``'homography (0,0)'`` through
            ``'homography (2,2)'`` as returned by :func:`pyneon.video.find_homographies`.
        %(max_gap_ms)s
        overwrite : bool, optional
            Only applicable if surface fixation columns already exist.
            If ``True``, overwrite existing columns. If ``False``, raise an error.
            Defaults to ``False``.
        %(inplace)s

        Returns
        -------
        %(events_or_none)s
        """
        inst = self if inplace else self.copy()
        data = inst.data

        if not overwrite and (
            "fixation x [surface coord]" in data.columns
            or "fixation y [surface coord]" in data.columns
        ):
            raise ValueError(
                "Events already contain fixation surface data. "
                "Use overwrite=True to overwrite existing columns."
            )

        required_cols = ["fixation x [px]", "fixation y [px]"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(
                f"Data must contain the following columns: {required_cols}"
            )

        data["fixation x [surface coord]"] = np.nan
        data["fixation y [surface coord]"] = np.nan

        event_ts = inst.start_ts
        homographies_data = homographies.interpolate(
            event_ts, float_kind="linear", max_gap_ms=max_gap_ms
        ).data

        h_cols = [f"homography ({i},{j})" for i in range(3) for j in range(3)]
        if not all(col in homographies_data.columns for col in h_cols):
            raise ValueError(f"Homographies data must contain columns: {h_cols}")

        homographies_data = homographies_data.dropna()

        x_col = data.columns.get_loc("fixation x [surface coord]")
        y_col = data.columns.get_loc("fixation y [surface coord]")

        for event_idx, ts in enumerate(event_ts):
            if ts not in homographies_data.index:
                continue

            h_row = homographies_data.loc[ts]
            if isinstance(h_row, pd.DataFrame):
                h_row = h_row.iloc[0]

            fix_vals = data.iloc[event_idx][required_cols].values
            if pd.isna(fix_vals).any():
                continue

            fix_points = np.asarray(fix_vals, dtype=np.float64).reshape(1, -1)
            H_flat = h_row[h_cols].values
            H = H_flat.reshape(3, 3)
            fix_trans = _apply_homography(fix_points, H)

            data.iat[event_idx, x_col] = fix_trans[:, 0]
            data.iat[event_idx, y_col] = fix_trans[:, 1]

        return None if inplace else inst
