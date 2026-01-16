from ast import literal_eval
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd

from .preprocess import (
    compute_azimuth_and_elevation,
    interpolate,
    interpolate_events,
    window_average,
)
from .tabular import BaseTabular
from .utils.doc_decorators import fill_doc
from .utils.variables import native_to_cloud_column_map, nominal_sampling_rates

if TYPE_CHECKING:
    from .events import Events


def _load_native_stream_data(raw_file: Path) -> tuple[pd.DataFrame, list[Path]]:
    """
    Directly load native Pupil Neon stream data from .raw, .time, and .dtype files.
    """
    rec_dir = raw_file.parent.resolve()
    time_file = raw_file.with_suffix(".time")
    dtype_file = next(rec_dir.glob(f"{raw_file.name[:3]}*.dtype"), None)
    if not time_file.is_file():
        raise FileNotFoundError(f"Missing .time file {time_file.name} in {rec_dir}")
    if dtype_file is None:
        raise FileNotFoundError(f"Missing .dtype file for {raw_file.name} in {rec_dir}")
    files = [raw_file, time_file, dtype_file]

    # Read timestamps
    ts = np.fromfile(time_file, dtype=np.int64)
    # Read data in the correct dtype
    dtype = np.dtype(literal_eval(dtype_file.read_text()))
    raw = np.fromfile(raw_file, dtype=dtype)
    if ts.shape[0] != raw.shape[0]:
        raise ValueError(
            f"Timestamp ({ts.shape[0]}) and data ({raw.shape[0]}) lengths do not match."
        )

    # Drop "timestamp_ns" field from raw data if present
    if "timestamp_ns" in raw.dtype.names:
        raw = raw[[n for n in raw.dtype.names if n != "timestamp_ns"]]

    # Create DataFrame with timestamps as index
    data = pd.DataFrame(raw, index=ts)
    data.index.name = "timestamp [ns]"

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

    # Concatenate worn data if loading gaze stream
    if "gaze x [px]" in data.columns:
        try:
            worn_dtype = np.dtype(literal_eval((rec_dir / "worn.dtype").read_text()))
            worn = np.fromfile(
                raw_file.with_name(raw_file.name.replace("gaze", "worn")),
                dtype=worn_dtype,
            )["worn"]
            data["worn"] = worn.astype(np.int8)
            compute_azimuth_and_elevation(data)
        except Exception as e:
            warn(f"Could not load 'worn' data for gaze stream: {e}")

    return data, files


def _infer_stream_type(data: pd.DataFrame) -> str:
    """
    Infer stream type based on presence of specific columns.
    If multiple or no matches found, return "custom".
    """
    col_map = {
        "gaze x [px]": "gaze",
        "pupil diameter left [mm]": "eye_states",
        "gyro x [deg/s]": "imu",
    }
    types = {col_map[c] for c in data.columns if c in col_map}
    return types.pop() if len(types) == 1 else "custom"


class Stream(BaseTabular):
    """
    Container for continuous data streams (gaze, eye states, IMU).

    Data is indexed by timestamps in nanoseconds.

    Parameters
    ----------
    source : pandas.DataFrame or pathlib.Path or str
        Source of the stream data. Can be either:

        - :class:`pandas.DataFrame`: Must contain a ``timestamp [ns]`` column or index.
        - :class:`pathlib.Path` or :class:`str`: Path to a stream data file. Supported file formats:

          - ``.csv``: Pupil Cloud format CSV file
          - ``.raw``: Native format (requires ``.time`` and ``.dtype`` files in the same directory)

        Note: Native format columns are automatically renamed to Pupil Cloud
        format for consistency. For example, ``gyro_x`` -> ``gyro x [deg/s]``.

    Attributes
    ----------
    file : pathlib.Path or None
        Path to the source file(s). ``None`` if initialized from DataFrame.
    data : pandas.DataFrame
        Stream data with ``timestamp [ns]`` as index.
    type : {"gaze", "eye_states", "imu", "custom"}
        Inferred stream type based on data columns.

    Examples
    --------
    Load from Pupil Cloud CSV:

    >>> gaze = Stream("gaze.csv")

    Load from native format:

    >>> gaze = Stream("gaze ps1.raw") # Or "gaze_200hz.raw"

    Create from DataFrame:

    >>> df = pd.DataFrame({"timestamp [ns]": [...], "gaze x [px]": [...]})
    >>> gaze = Stream(df)
    """

    def __init__(self, source: pd.DataFrame | Path | str):
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            if not source.is_file():
                raise FileNotFoundError(f"File does not exist: {source}")
            if source.suffix == ".csv":  # Path to Pupil Cloud CSV file
                self.file = source
                data = pd.read_csv(source, index_col="timestamp [ns]")
            elif source.suffix == ".raw":
                data, self.file = _load_native_stream_data(source)
            else:
                raise ValueError(
                    "Unsupported file format. Only .csv and .raw are supported."
                )
        else:  # pd.DataFrame
            data = source.copy(deep=True)
            self.file = None

        if data.index.name != "timestamp [ns]":
            if "timestamp [ns]" in data.columns:
                data = data.set_index("timestamp [ns]")
            else:
                raise ValueError("Data does not contain a valid timestamp column")
        data.sort_index(inplace=True)

        super().__init__(data)
        self.type = _infer_stream_type(self.data)

    def __getitem__(self, index: str) -> pd.Series:
        if index not in self.data.columns:
            raise KeyError(f"Column '{index}' not found in the stream data.")
        return self.data[index]

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps of the stream in nanoseconds."""
        return self.data.index.to_numpy()

    @property
    def ts(self) -> np.ndarray:
        """Alias for ``timestamps``."""
        return self.timestamps

    @property
    def first_ts(self) -> int:
        """First timestamp of the stream."""
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Last timestamp of the stream."""
        return int(self.ts[-1])

    @property
    def ts_diff(self) -> np.ndarray:
        """Difference between consecutive timestamps."""
        return np.diff(self.ts)

    @property
    def times(self) -> np.ndarray:
        """Timestamps converted to seconds relative to stream start."""
        return (self.ts - self.first_ts) / 1e9

    @property
    def duration(self) -> float:
        """Duration of the stream in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def sampling_freq_effective(self) -> float:
        """Effective sampling frequency of the stream."""
        return len(self.data) / self.duration

    @property
    def sampling_freq_nominal(self) -> Optional[int]:
        """
        Nominal sampling frequency in Hz as specified by Pupil Labs
        (see https://pupil-labs.com/products/neon/specs).
        ``None`` for custom or unknown stream types.
        """
        return nominal_sampling_rates.get(self.type, None)

    @property
    def is_uniformly_sampled(self) -> bool:
        """Whether the stream is uniformly sampled."""
        return np.allclose(self.ts_diff, self.ts_diff[0])

    def time_to_ts(self, time: Number | np.ndarray) -> np.ndarray:
        """Convert relative time(s) in seconds to closest timestamp(s) in nanoseconds."""
        time = np.array([time])
        return np.array([self.ts[np.absolute(self.times - t).argmin()] for t in time])

    @fill_doc
    def crop(
        self,
        tmin: Optional[Number] = None,
        tmax: Optional[Number] = None,
        by: Literal["timestamp", "time", "row"] = "timestamp",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Crop data to a specific time range based on timestamps,
        relative times since start, or row numbers.

        Parameters
        ----------
        tmin : numbers.Number, optional
            Start timestamp/time/row to crop the data to. If ``None``,
            the minimum timestamp/time/row in the data is used. Defaults to ``None``.
        tmax : numbers.Number, optional
            End timestamp/time/row to crop the data to. If ``None``,
            the maximum timestamp/time/row in the data is used. Defaults to ``None``.
        by : "timestamp" or "time" or "row", optional
            Whether tmin and tmax are Unix timestamps in nanoseconds
            OR relative times in seconds OR row numbers of the stream data.
            Defaults to "timestamp".

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s
        """
        if tmin is None and tmax is None:
            raise ValueError("At least one of `tmin` or `tmax` must be provided")
        if by == "timestamp":
            t = self.ts
        elif by == "time":
            t = self.times
        else:
            t = np.arange(len(self))
        tmin = t.min() if tmin is None else tmin
        tmax = t.max() if tmax is None else tmax
        # tmin and tmax should be positive numbers
        if tmin < 0 or tmax < 0:
            raise ValueError("Crop bounds must be non-negative")
        mask = (t >= tmin) & (t <= tmax)
        if not mask.any():
            raise ValueError("No data found in the specified time range")
        inst = self if inplace else self.copy()
        inst.data = self.data[mask].copy()
        return None if inplace else inst

    @fill_doc
    def restrict(
        self,
        other: "Stream",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Temporally crop the stream to the range of timestamps in another stream.
        Equivalent to ``crop(other.first_ts, other.last_ts)``.

        Parameters
        ----------
        other : Stream
            The other stream whose timestamps are used to restrict the data.

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s
        """
        return self.crop(other.first_ts, other.last_ts, by="timestamp", inplace=inplace)

    @fill_doc
    def interpolate(
        self,
        new_ts: Optional[np.ndarray] = None,
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Interpolate the stream to new timestamps.

        Data columns of float type are interpolated using ``float_kind``,
        while other columns use ``other_kind``. This distinction allows
        for appropriate interpolation methods based on data type.

        Parameters
        ----------
        new_ts : numpy.ndarray, optional
            An array of new timestamps (in nanoseconds) at which to evaluate
            the interpolant. If ``None`` (default), new and equally-spaced timestamps
            are generated according to :attr:`sampling_freq_nominal`.

        %(interp_kwargs)s

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s

        Notes
        -----
        - If ``new_ts`` contains timestamps outside the range of ``self.ts``,
          the corresponding rows will contain NaN.
        """
        # If new_ts is not provided, generate a evenly spaced array of timestamps
        if new_ts is None:
            if self.sampling_freq_nominal is None:
                raise ValueError(
                    "The nominal sampling frequency of the stream is not specified, "
                    "please specify new_ts manually."
                )
            step_size = int(1e9 / self.sampling_freq_nominal)
            new_ts = np.arange(self.first_ts, self.last_ts, step_size, dtype=np.int64)
            assert new_ts[0] == self.first_ts
            assert np.allclose(np.diff(new_ts), step_size)

        inst = self if inplace else self.copy()
        inst.data = interpolate(new_ts, self.data, float_kind, other_kind)
        return None if inplace else inst

    @fill_doc
    def annotate_events(
        self, events: "Events", overwrite: bool = False, inplace: bool = False
    ) -> Optional["Stream"]:
        """
        Annotate stream data with event IDs based on event time intervals.

        Parameters
        ----------
        events : Events
            Events object containing the events to annotate.
            The events must have a valid ``id_name`` attribute,
            as well as ``start timestamp [ns]`` and ``end timestamp [ns]`` columns.
        overwrite : bool, optional
            If ``True``, overwrite existing event ID annotations in the stream data.
            Defaults to ``False``.

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s

        Raises
        ------
        ValueError
            If no event ID column is known for the Events instance.
        KeyError
            If the expected event ID column is not found in the Events data.
        """
        id_name = events.id_name
        if id_name is None:
            raise ValueError(
                "Cannot annotate events as no event ID column is known for the Events instance."
            )
        if id_name not in events.data.columns:
            raise KeyError(
                f"Events data does not contain the expected ID column: {id_name}"
            )
        if not overwrite and id_name in self.data.columns:
            raise ValueError(
                f"Stream data already contains a column named '{id_name}'. "
                "Use overwrite=True to overwrite existing annotations."
            )
        inst = self if inplace else self.copy()

        # Initialize nullable integer column
        inst.data[id_name] = pd.Series(pd.NA, dtype="Int64")

        # Vectorized annotation using IntervalIndex
        intervals = pd.IntervalIndex.from_arrays(
            events.data["start timestamp [ns]"],
            events.data["end timestamp [ns]"],
            closed="both",
        )
        # Get, for each timestamp, the index of the event it falls into (-1 if none)
        event_idx = intervals.get_indexer(inst.data.index)
        # Assign event IDs where applicable
        valid = event_idx != -1
        inst.data.loc[valid, id_name] = events.data.iloc[event_idx[valid]][
            id_name
        ].values.astype("Int64")

        return None if inplace else inst

    @fill_doc
    def interpolate_events(
        self,
        events: "Events",
        buffer: Number | tuple[Number, Number] = 0.05,
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Interpolate data in the duration of events in the stream data.
        Similar to :func:`mne.preprocessing.eyetracking.interpolate_blinks`.

        Parameters
        ----------
        events : Events
            Events object containing the events to interpolate.
            The events must have ``start timestamp [ns]`` and
            ``end timestamp [ns]`` columns.
        buffer : numbers.Number or , optional
            The time before and after an event (in seconds) to consider invalid.
            If a single number is provided, the same buffer is applied
            to both before and after the event.
            Defaults to 0.05.

        %(interp_kwargs)s

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s

        Examples
        --------
        Interpolate eye states data during blinks with a 50 ms buffer before and after:

        >>> eye_states = eye_states.interpolate_events(blinks, buffer=0.05)
        """
        inst = self if inplace else self.copy()
        inst.data = interpolate_events(
            self.data,
            events,
            buffer,
            float_kind=float_kind,
            other_kind=other_kind,
        )
        return None if inplace else inst

    @fill_doc
    def window_average(
        self,
        new_ts: np.ndarray,
        window_size: Optional[int] = None,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Take the average over a time window to obtain smoothed data at new timestamps.

        Parameters
        ----------
        new_ts : numpy.ndarray
            An array of new timestamps (in nanoseconds) at which to evaluate the
            averaged signal. Must be coarser than the source
            sampling, i.e.:

            >>> np.median(np.diff(new_ts)) > np.median(np.diff(data.index))

        window_size : int, optional
            The size of the time window (in nanoseconds)
            over which to compute the average around each new timestamp.
            If ``None`` (default), the window size is set to the median interval
            between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
            The window size must be larger than the median interval between the original data timestamps,
            i.e., ``window_size > np.median(np.diff(data.index))``.

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s
        """
        inst = self if inplace else self.copy()
        inst.data = window_average(new_ts, self.data, window_size)
        return None if inplace else inst

    @fill_doc
    def compute_azimuth_and_elevation(
        self,
        method: Literal["linear"] = "linear",
        overwrite: bool = False,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Compute gaze azimuth and elevation angles (in degrees)
        based on gaze pixel coordinates and append them to the stream data.

        Parameters
        ----------
        method : {"linear"}, optional
            Method to compute gaze angles. Defaults to "linear".
        overwrite : bool, optional
            Only applicable if azimuth and elevation columns already exist.
            If ``True``, overwrite existing columns. If ``False``, raise an error.
            Defaults to ``False``.

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s

        Raises
        ------
        ValueError
            If required gaze columns are not present in the data.
        """
        if not overwrite and (
            "azimuth [deg]" in self.data.columns
            or "elevation [deg]" in self.data.columns
        ):
            raise ValueError(
                "Stream data already contains azimuth and/or elevation columns. "
                "Use overwrite=True to overwrite existing columns."
            )
        inst = self if inplace else self.copy()
        compute_azimuth_and_elevation(inst.data, method=method)
        return None if inplace else inst

    @fill_doc
    def concat(
        self,
        other: "Stream",
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Concatenate additional columns from another Stream to this Stream.
        The other Stream will be interpolated to the timestamps of this Stream
        to achieve temporal alignment. See :meth:`interpolate` for details.

        Parameters
        ----------
        other : Stream
            The other stream to concatenate.

        %(interp_kwargs)s

        %(inplace)s

        Returns
        -------
        %(stream_or_none)s
        """
        # Interpolate other to self timestamps if needed
        if not np.array_equal(self.ts, other.ts):
            other = other.interpolate(self.ts, float_kind, other_kind, inplace=False)

        other_data = other.data.copy()

        # Check for overlapping columns
        overlap_cols = self.columns.intersection(other_data.columns)
        if len(overlap_cols) > 0:
            warn(
                f"Overlapping columns detected: {list(overlap_cols)}. "
                "Keeping values from the first dataframe only.",
                UserWarning,
            )
            # Drop overlapping columns from other
            other_data.drop(columns=overlap_cols, inplace=True)

        inst = self if inplace else self.copy()
        inst.data = pd.concat([inst.data, other_data], axis=1)
        return None if inplace else inst
