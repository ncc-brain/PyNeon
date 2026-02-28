from ast import literal_eval
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocess import (
    compute_azimuth_and_elevation,
    interpolate,
    interpolate_events,
    window_average,
)
from .tabular import BaseTabular
from .utils import _apply_homography, _validate_df_columns
from .utils.doc_decorators import fill_doc
from .utils.variables import native_to_cloud_column_map, nominal_sampling_rates

if TYPE_CHECKING:
    from .events import Events


def _apply_homographies_on_gaze(
    gaze: "Stream",
    homographies: "Stream",
    max_gap_ms: Number = 500,
    overwrite: bool = False,
) -> None:
    """
    Apply homographies to gaze points.

    Since homographies are estimated per video frame and might not be available
    for every frame, they need to be resampled/interpolated to the timestamps of the
    gaze data before application. Users can control the extent of interpolation using
    the ``max_gap_ms`` parameter to avoid applying homographies over large gaps.

    This function operates in-place and modifies the `gaze` Stream by adding two new columns:
    'gaze x [surface coord]' and 'gaze y [surface coord]'.

    Parameters
    ----------
    gaze : Stream
        Stream containing gaze points with columns 'gaze x [px]' and 'gaze y [px]'.
    homographies : Stream
        Stream containing homography matrices with columns 'homography (0,0)' through
        'homography (2,2)' as returned by :func:`pyneon.video.find_homographies`.
    %(max_gap_ms_param)s
    overwrite : bool, optional
        If True, overwrite existing surface coordinate columns. Defaults to False.

    Returns
    -------
    None
        This function modifies the gaze Stream in-place.
    """
    gaze_data = gaze.data
    if not overwrite and (
        "gaze x [surface coord]" in gaze_data.columns
        or "gaze y [surface coord]" in gaze_data.columns
    ):
        raise ValueError(
            "Stream already contains gaze on surface data. "
            "Use overwrite=True to overwrite existing columns."
        )

    required_cols = ["gaze x [px]", "gaze y [px]"]
    if not all(col in gaze_data.columns for col in required_cols):
        raise ValueError(f"Data must contain the following columns: {required_cols}")

    gaze_data["gaze x [surface coord]"] = np.nan
    gaze_data["gaze y [surface coord]"] = np.nan

    # Interpolate homographies to gaze timestamps
    homographies_data = homographies.interpolate(
        gaze.ts, float_kind="linear", max_gap_ms=max_gap_ms
    ).data
    # Exclude all rows where homography is NaN after interpolation
    homographies_data = homographies_data.dropna()

    # Extract homography column names in order
    h_cols = [f"homography ({i},{j})" for i in range(3) for j in range(3)]

    # Validate gaze data has required columns
    _validate_df_columns(gaze_data, ["gaze x [px]", "gaze y [px]"], df_name="gaze")

    # Validate that all homography columns exist
    _validate_df_columns(homographies_data, h_cols, df_name="homographies")

    # Apply homographies to each gaze point
    for ts in tqdm(
        homographies_data.index, desc="Applying homographies to gaze points"
    ):
        # Get gaze point(s) at this timestamp
        gaze_row = gaze_data.loc[ts]
        gaze_vals = gaze_row[["gaze x [px]", "gaze y [px]"]].values

        # Skip if gaze coordinates are NaN
        if pd.isna(gaze_vals).any():
            continue

        # Convert to numpy array to ensure compatibility with _apply_homography
        gaze_points = np.asarray(gaze_vals, dtype=np.float64).reshape(1, -1)

        # Reconstruct homography matrix from the 9 columns
        H_flat = homographies_data.loc[ts, h_cols].values
        H = H_flat.reshape(3, 3)

        # Apply homography transformation
        gaze_trans = _apply_homography(gaze_points, H)
        gaze_data.loc[ts, "gaze x [surface coord]"] = gaze_trans[:, 0]
        gaze_data.loc[ts, "gaze y [surface coord]"] = gaze_trans[:, 1]


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
        "marker id": "marker detections",
        "homography (0,0)": "homographies",
    }
    types = {col_map[c] for c in data.columns if c in col_map}
    return types.pop() if len(types) == 1 else "custom"


class Stream(BaseTabular):
    """
    Container for continuous data streams (gaze, eye states, IMU).

    Data is indexed by timestamps in nanoseconds (``timestamp [ns]``).

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
    type : str
        Inferred stream type based on data columns.

    Examples
    --------
    Load from Pupil Cloud CSV:

    >>> from pyneon import Stream
    >>> gaze = Stream("gaze.csv")

    Load from native format:

    >>> gaze = Stream("gaze ps1.raw") # Or "gaze_200hz.raw"

    Create from DataFrame:

    >>> df = pd.DataFrame({"timestamp [ns]": [...], "gaze x [px]": [...]})
    >>> gaze = Stream(df)
    """

    file: Optional[Path]
    data: pd.DataFrame
    type: str

    def __init__(self, source: pd.DataFrame | Path | str):
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            if not source.is_file():
                raise FileNotFoundError(f"{source} does not exist")
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

    def __repr__(self) -> str:
        return f"""Stream type: {self.type}
Number of samples: {len(self)}
First timestamp: {self.first_ts}
Last timestamp: {self.last_ts}
Uniformly sampled: {self.is_uniformly_sampled}
Duration: {self.duration:.2f} seconds
Effective sampling frequency: {self.sampling_freq_effective:.2f} Hz
Nominal sampling frequency: {self.sampling_freq_nominal} Hz
Columns: {list(self.data.columns)}
"""

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps of the stream in nanoseconds.

        Returns
        -------
        numpy.ndarray
            Array of timestamps in nanoseconds (Unix time).
        """
        return self.data.index.to_numpy(dtype=np.int64)

    @property
    def ts(self) -> np.ndarray:
        """Alias for :attr:`timestamps`.

        Returns
        -------
        numpy.ndarray
            Array of timestamps in nanoseconds (Unix time).
        """
        return self.timestamps

    @property
    def first_ts(self) -> int:
        """First timestamp of the stream in nanoseconds.

        Returns
        -------
        int
            First timestamp in nanoseconds (Unix time).
        """
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Last timestamp of the stream in nanoseconds.

        Returns
        -------
        int
            Last timestamp in nanoseconds (Unix time).
        """
        return int(self.ts[-1])

    @property
    def ts_diff(self) -> np.ndarray:
        """Difference between consecutive timestamps.

        Returns
        -------
        numpy.ndarray
            Array of time differences in nanoseconds.
        """
        return np.diff(self.ts)

    @property
    def times(self) -> np.ndarray:
        """Timestamps converted to seconds relative to stream start.

        Returns
        -------
        numpy.ndarray
            Array of times in seconds, starting from 0.
        """
        return (self.ts - self.first_ts) / 1e9

    @property
    def duration(self) -> float:
        """Duration of the stream in seconds.

        Returns
        -------
        float
            Total duration from first to last timestamp in seconds.
        """
        return float(self.times[-1] - self.times[0])

    @property
    def sampling_freq_effective(self) -> float:
        """Effective/empirical sampling frequency of the stream in Hz.

        Returns
        -------
        float
            Effective sampling frequency in Hz.
        """
        return float(1e9 / self.ts_diff.mean())

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
        """Whether the stream is uniformly sampled.

        Returns
        -------
        bool
            True if all consecutive timestamp differences are approximately equal.
        """
        return np.allclose(self.ts_diff, self.ts_diff[0])

    def time_to_ts(self, time: Number | np.ndarray) -> np.ndarray:
        """Convert relative time(s) in seconds to the closest timestamp(s) in nanoseconds.

        Parameters
        ----------
        time : numbers.Number or numpy.ndarray
            Time(s) in seconds relative to stream start.

        Returns
        -------
        numpy.ndarray
            Corresponding timestamp(s) in nanoseconds.
        """
        time = np.array([time])
        return np.array([self.ts[np.absolute(self.times - t).argmin()] for t in time])

    @fill_doc
    def crop(
        self,
        tmin: Optional[Number] = None,
        tmax: Optional[Number] = None,
        by: Literal["timestamp", "time", "sample"] = "timestamp",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Extract a subset of stream data within a specified temporal range.

        The ``by`` parameter determines how ``tmin`` and ``tmax`` are interpreted:
        - ``"timestamp"``: Absolute Unix timestamps in nanoseconds
        - ``"time"``: Relative time in seconds from the stream's first sample
        - ``"sample"``: Zero-based sample indices

        Both bounds are inclusive. If either bound is omitted, it defaults to the
        stream's natural boundary (earliest or latest sample).

        Parameters
        ----------
        tmin : numbers.Number, optional
            Lower bound of the range to extract (inclusive). If ``None``,
            starts from the stream's beginning. Defaults to ``None``.
        tmax : numbers.Number, optional
            Upper bound of the range to extract (inclusive). If ``None``,
            extends to the stream's end. Defaults to ``None``.
        by : {"timestamp", "time", "sample"}, optional
            Unit used to interpret ``tmin`` and ``tmax``. Defaults to ``"timestamp"``.

        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

        Raises
        ------
        ValueError
            If both ``tmin`` and ``tmax`` are ``None``, if bounds are negative,
            or if no data falls within the specified range.

        Examples
        --------
        Crop to the first 0.5 seconds of data:

        >>> stream_500ms = stream.crop(tmin=0, tmax=0.5, by="time")

        Crop using absolute timestamps:

        >>> cropped = stream.crop(tmin=start_ts, tmax=end_ts, by="timestamp")

        Extract samples 100 through 200:

        >>> samples = stream.crop(tmin=100, tmax=200, by="sample")
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
        Align this stream's temporal range to match another stream.

        This method crops the data to include only samples between the first and
        last timestamps of the reference stream. It is equivalent to calling
        ``crop(tmin=other.first_ts, tmax=other.last_ts, by="timestamp")``.

        Useful for ensuring temporal alignment across multiple data streams,
        particularly when streams have different start or end times.

        Parameters
        ----------
        other : Stream
            Reference stream whose temporal boundaries define the cropping range.

        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

        Examples
        --------
        Align IMU data to match the temporal extent of gaze data:

        >>> imu_aligned = imu.restrict(gaze)
        """
        return self.crop(other.first_ts, other.last_ts, by="timestamp", inplace=inplace)

    @fill_doc
    def interpolate(
        self,
        new_ts: Optional[np.ndarray] = None,
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        max_gap_ms: Optional[Number] = 500,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Interpolate the stream to new timestamps.
        Useful for temporal synchronization (e.g., stream-to-stream, stream-to-video) or
        resampling to a uniform rate.

        Data columns of float type are interpolated using the method specified by ``float_kind``,
        while other columns use the method specified by ``other_kind``. This distinction allows
        for appropriate interpolation methods based on data type.

        Parameters
        ----------
        new_ts : numpy.ndarray, optional
            Target timestamps (in nanoseconds) for the resampled data. If ``None``,
            timestamps are auto-generated at uniform intervals based on
            :attr:`sampling_freq_nominal`. Defaults to ``None``.

        %(interp_kind_params)s
        %(max_gap_ms_param)s
        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

        Notes
        -----
        - Timestamps in ``new_ts`` that fall outside the original data range
          will have NaN values in the interpolated stream.
        - Column data types are preserved after interpolation.
        - Uses `scipy.interpolate.interp1d` internally.

        Examples
        --------
        Interpolate gaze data to uniform 200 Hz sampling:

        >>> gaze_uniform = gaze.interpolate()

        Align gaze data to IMU timestamps:

        >>> gaze_on_imu = gaze.interpolate(new_ts=imu.ts)
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
        inst.data = interpolate(new_ts, self.data, float_kind, other_kind, max_gap_ms)
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
            Events instance containing the events to annotate.
            The events must have a valid :attr:`id_name` attribute,
            as well as ``start timestamp [ns]`` and ``end timestamp [ns]`` columns.
        overwrite : bool, optional
            If ``True``, overwrite existing event ID annotations in the stream data.
            Defaults to ``False``.

        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

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
        max_gap_ms: Optional[Number] = None,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Interpolate data during the duration of events in the stream data.
        Particularly useful for repairing blink artifacts in eye states or gaze data.
        Similar to :func:`mne.preprocessing.eyetracking.interpolate_blinks`.

        Parameters
        ----------
        events : Events
            Events instance containing the events to interpolate.
            The events must have ``start timestamp [ns]`` and
            ``end timestamp [ns]`` columns.
        buffer : numbers.Number or tuple[numbers.Number, numbers.Number], optional
            The time before and after an event (in seconds) to consider invalid.
            If a single number is provided, the same buffer is applied
            to both before and after the event.
            Defaults to 0.05.

        %(interp_kind_params)s
        %(max_gap_ms_param)s
        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

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
            max_gap_ms=max_gap_ms,
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

        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s
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
        Compute gaze azimuth and elevation angles (in degrees) from pixel coordinates
        based on gaze pixel coordinates and append them to the stream data.

        The stream data must contain the required gaze columns:
        ``gaze x [px]`` and ``gaze y [px]``.


        Parameters
        ----------
        method : {"linear"}, optional
            Method to compute gaze angles. Defaults to "linear".
        overwrite : bool, optional
            Only applicable if azimuth and elevation columns already exist.
            If ``True``, overwrite existing columns. If ``False``, raise an error.
            Defaults to ``False``.

        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

        Raises
        ------
        ValueError
            If required gaze columns are not present in the data.
        """
        inst = self if inplace else self.copy()
        compute_azimuth_and_elevation(inst.data, method, overwrite)
        return None if inplace else inst

    @fill_doc
    def apply_homographies(
        self,
        homographies: "Stream",
        max_gap_ms: Number = 500,
        overwrite: bool = False,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Compute gaze locations in surface coordinates using provided homographies
        based on gaze pixel coordinates and append them to the stream data.

        Since homographies are estimated per video frame and might not be available
        for every frame, they need to be resampled/interpolated to the timestamps of the
        gaze data before application.

        The stream data must contain the required gaze columns:
        ``gaze x [px]`` and ``gaze y [px]``.
        The output stream will contain two new columns:
        ``gaze x [surface coord]`` and ``gaze y [surface coord]``.

        Parameters
        ----------
        %(homographies)s
            Returned by :func:`pyneon.find_homographies`.
        %(max_gap_ms_param)s
        overwrite : bool, optional
            Only applicable if surface gaze columns already exist.
            If ``True``, overwrite existing columns. If ``False``, raise an error.
            Defaults to ``False``.
        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s
        """
        inst = self if inplace else self.copy()
        _apply_homographies_on_gaze(inst, homographies, max_gap_ms, overwrite)
        return None if inplace else inst

    @fill_doc
    def concat(
        self,
        other: "Stream",
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        max_gap_ms: Optional[Number] = 500,
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

        %(interp_kind_params)s
        %(max_gap_ms_param)s
        %(inplace_param)s

        Returns
        -------
        %(stream_or_none_returns)s

        Notes
        -----
        To concatenate multiple :class:`Stream` throughout the recording, you can also use
        :meth:`Recording.concat_streams()`.
        """
        # Interpolate other to self timestamps if needed
        if not np.array_equal(self.ts, other.ts):
            other = other.interpolate(
                self.ts, float_kind, other_kind, max_gap_ms=max_gap_ms, inplace=False
            )

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
