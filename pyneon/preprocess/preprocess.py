import pandas as pd
import numpy as np
from tqdm import tqdm

from pandas.api.types import (
    is_float_dtype, is_numeric_dtype, is_bool_dtype,
    is_object_dtype, is_string_dtype, is_categorical_dtype
)

from numbers import Number
from typing import TYPE_CHECKING, Optional
from scipy.interpolate import interp1d

from ..utils import _check_data

if TYPE_CHECKING:
    from ..recording import Recording
    from ..events import Events


def interpolate(
    new_ts: np.ndarray,
    data: pd.DataFrame,
    float_kind: str = "linear",
    other_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Interpolate a data stream to a new set of timestamps.

    Parameters
    ----------
    new_ts : numpy.ndarray
        An array of new timestamps (in nanoseconds)
        at which to evaluate the interpolant.
    data : pandas.DataFrame
        Source data to interpolate. Must have a monotonically increasing
        index named ``timestamp [ns]``.
    float_kind : str, optional
        Kind of interpolation applied on columns of ``float`` type,
        For details see :class:`scipy.interpolate.interp1d`.
        Defaults to ``"linear"``.
    other_kind : str, optional
        Kind of interpolation applied on columns of other types,
        For details see :class:`scipy.interpolate.interp1d`.
        Defaults to ``"nearest"``.

    Returns
    -------
    pandas.DataFrame
        Interpolated data with the same columns and dtypes as ``data``
        and indexed by ``new_ts``.
    """
    _check_data(data)

    if isinstance(data.index, pd.DatetimeIndex):
        x = data.index.view("int64")
    else:
        x = data.index.astype("int64")

    new_ts = np.sort(new_ts).astype("int64")
    new_data = pd.DataFrame(index=new_ts, columns=data.columns)

    for col in data.columns:
        s = data[col]

        # 1) floats -> interp with float_kind
        if is_float_dtype(s):
            y = s.to_numpy(dtype=float, copy=False)
            f = interp1d(x, y, kind=float_kind, bounds_error=False)
            vals = f(new_ts)

        # 2) other numeric / bool -> interp with other_kind
        elif is_numeric_dtype(s) or is_bool_dtype(s):
            y = s.to_numpy(dtype=float, copy=False)
            f = interp1d(x, y, kind=other_kind, bounds_error=False)
            vals = f(new_ts)

        # 3) strings / objects / categoricals -> nearest neighbor (no numeric interp!)
        elif is_object_dtype(s) or is_string_dtype(s) or is_categorical_dtype(s):
            # Build a Series with the original (numeric) time index
            ser = pd.Series(s.values, index=x).sort_index()
            # Collapse duplicate timestamps (keep the last; choose 'first' if you prefer)
            if ser.index.has_duplicates:
                # either of these is fine:
                # ser = ser.groupby(level=0).last()
                ser = ser[~ser.index.duplicated(keep="last")]
            # Now nearest works because the index is unique & monotonic
            vals = ser.reindex(new_ts, method="nearest").to_numpy()


        else:
            # fallback: treat like other numeric
            y = s.to_numpy(dtype=float, copy=False)
            f = interp1d(x, y, kind=other_kind, bounds_error=False)
            vals = f(new_ts)

        # cast back if numeric/bool; leave objects/categories as-is
        try:
            new_data[col] = vals.astype(s.dtype, copy=False) if (is_numeric_dtype(s) or is_bool_dtype(s)) else vals
        except (TypeError, ValueError):
            new_data[col] = vals  # safest fallback

    new_data.index.name = data.index.name
    return new_data


def interpolate_events(
    data: pd.DataFrame,
    events: "Events",
    buffer: Number | tuple[Number, Number] = 0.05,
    float_kind: str = "linear",
    other_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Interpolate data in the duration of events in the stream data.
    Similar to :func:`mne.preprocessing.eyetracking.interpolate_blinks`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to interpolate. Must have a monotonically increasing
        index named ``timestamp [ns]``.
    events : Events
        Events object containing the events to interpolate.
        The events must have ``start timestamp [ns]`` and
        ``end timestamp [ns]`` columns.
    buffer : numbers.Number or , optional
        The time before and after an event (in seconds) to consider invalid.
        If a single number is provided, the same buffer is applied
        to both before and after the event.
        Defaults to 0.05.
    float_kind : str, optional
        Kind of interpolation applied on columns of ``float`` type,
        For details see :class:`scipy.interpolate.interp1d`.
        Defaults to ``"linear"``.
    other_kind : str, optional
        Kind of interpolation applied on columns of other types,
        For details see :class:`scipy.interpolate.interp1d`.
        Defaults to ``"nearest"``.

    Returns
    -------
    pandas.DataFrame
        Interpolated data with the same columns and dtypes as ``data``
        and indexed by ``data.index``.
    """
    _check_data(data)

    # Make a (2, n_blink) matrix of blink start and end timestamps
    event_times = np.array(
        [
            events.start_ts,
            events.end_ts,
        ]
    )
    # Add buffer to the blink times (start_ts - buffer, end_ts + buffer)
    if isinstance(buffer, Number):
        buffer = (buffer, buffer)
    event_times[0] -= int(buffer[0] * 1e9)
    event_times[1] += int(buffer[1] * 1e9)

    data_ts = data.index.values

    # Find data in the blink times
    mask = np.zeros(data_ts.shape, dtype=bool)
    for blink in event_times.T:
        mask |= (data_ts >= blink[0]) & (data_ts <= blink[1])
    new_data = data[~mask]

    # Interpolate to original timestamps
    new_data = interpolate(
        data_ts,
        new_data,
        float_kind=float_kind,
        other_kind=other_kind,
    )
    return new_data


def window_average(
    new_ts: np.ndarray, data: pd.DataFrame, window_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Take the average over a time window to obtain smoothed data at new timestamps.

    Parameters
    ----------
    new_ts : numpy.ndarray
        An array of new timestamps (in nanoseconds) at which to evaluate the
        averaged signal. Must be coarser than the source
        sampling, i.e.:

        >>> np.median(np.diff(new_ts)) > np.median(np.diff(data.index))

    data : pandas.DataFrame
        Source data to apply window average to. Must have a monotonically increasing
        index named ``timestamp [ns]``.
    window_size : int, optional
        The size of the time window (in nanoseconds)
        over which to compute the average around each new timestamp.
        If ``None`` (default), the window size is set to the median interval
        between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
        The window size must be larger than the median interval between the original data timestamps,
        i.e., ``window_size > np.median(np.diff(data.index))``.

    Returns
    -------
    pandas.DataFrame
        Data with window average applied, carrying the same columns and
        dtypes as ``data`` and indexed by ``new_ts``. Non-float columns are rounded back to their
        original integer type after averaging.
    """

    _check_data(data)
    new_ts = np.sort(new_ts).astype("int64")

    # ------------------------------------------------------------------ checks
    original_diff = np.median(np.diff(data.index))
    new_diff = np.median(np.diff(new_ts))
    if new_diff < original_diff:
        raise ValueError("new_ts must be down-sampled relative to the data.")
    if window_size is None:
        window_size = int(new_diff)
    if window_size < original_diff:
        raise ValueError("window_size must exceed original sample spacing.")

    # Convert the int64‑ns index into a DatetimeIndex because pandas'
    # rolling(time‑window) API works on Datetime/Timedelta indices.
    df = data.copy()
    df.index = pd.to_datetime(df.index, unit="ns")
    target_idx = pd.to_datetime(new_ts, unit="ns")

    df = df.reindex(df.index.union(target_idx)).sort_index()

    win_str = f"{window_size}ns"
    rolled = df.rolling(win_str, center=True, min_periods=1).mean()

    new_data = rolled.loc[target_idx]

    non_float = data.select_dtypes(exclude="float").columns
    new_data[non_float] = new_data[non_float].round().astype(data[non_float].dtypes)

    new_data.index = new_ts
    new_data.index.name = data.index.name
    return new_data


_VALID_STREAMS = {"3d_eye_states", "eye_states", "gaze", "imu"}


def concat_streams(
    rec: "Recording",
    stream_names: str | list[str] = "all",
    sampling_freq: Number | str = "min",
    interp_float_kind: str = "linear",
    interp_other_kind: str = "nearest",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Concatenate data from different streams under common timestamps.
    Since the streams may have different timestamps and sampling frequencies,
    interpolation of all streams to a set of common timestamps is performed.
    The latest start timestamp and earliest last timestamp of the selected streams
    are used to define the common timestamps.

    Parameters
    ----------
    rec : Recording
        Recording object containing the streams to concatenate.
    stream_names : str or list of str
        Stream names to concatenate. If "all" (default), then all streams will be used.
        If a list, items must be in ``{"gaze", "imu", "eye_states"}``
        (``"3d_eye_states"``) is also tolerated as an alias for ``"eye_states"``).
    sampling_freq : numbers.Number or str, optional
        Sampling frequency of the concatenated streams.
        If numeric, the streams will be interpolated to this frequency.
        If "min" (default), the lowest nominal sampling frequency
        of the selected streams will be used.
        If "max", the highest nominal sampling frequency will be used.
    interp_float_kind : str, optional
        Kind of interpolation applied on columns of ``float`` type,
        Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
    interp_other_kind : str, optional
        Kind of interpolation applied on columns of other types.
        Defaults to ``"nearest"``.
    inplace : bool, optional
        Replace selected stream data with interpolated data during concatenation
        if``True``. Defaults to ``False``.

    Returns
    -------
    concat_data : pandas.DataFrame
        Concatenated data.
    """
    if isinstance(stream_names, str):
        if stream_names == "all":
            stream_names = list(_VALID_STREAMS)
        else:
            raise ValueError(
                "Invalid stream_names, must be 'all' or a list of stream names."
            )
    if len(stream_names) <= 1:
        raise ValueError("Must provide at least two streams to concatenate.")

    stream_names = [ch.lower() for ch in stream_names]
    # Check if all streams are valid
    if not all([ch in _VALID_STREAMS for ch in stream_names]):
        raise ValueError(f"Invalid stream name, can only one of {_VALID_STREAMS}")

    stream_info = pd.DataFrame(columns=["stream", "name", "sf", "first_ts", "last_ts"])
    print("Concatenating streams:")
    stream_map = {
        "gaze": ("gaze", "gaze"),
        "3d_eye_states": ("eye_states", "3d_eye_states"),
        "eye_states": ("eye_states", "3d_eye_states"),
        "imu": ("imu", "imu"),
    }
    for name in stream_names:
        attr, display_name = stream_map[name]
        stream_obj = getattr(rec, attr, None)
        if stream_obj is None:
            raise ValueError(f"Cannot load {display_name} data.")
        stream_info = pd.concat(
            [
                stream_info,
                pd.Series(
                    {
                        "stream": stream_obj,
                        "name": display_name,
                        "sf": stream_obj.sampling_freq_nominal,
                        "first_ts": stream_obj.first_ts,
                        "last_ts": stream_obj.last_ts,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )
        print(f"\t{display_name.capitalize()}")

    # Lowest sampling rate
    if sampling_freq == "min":
        sf = stream_info["sf"].min()
        sf_type = "lowest"
    elif sampling_freq == "max":
        sf = stream_info["sf"].max()
        sf_type = "highest"
    elif isinstance(sampling_freq, (int, float)):
        sf = sampling_freq
        sf_type = "customized"
    else:
        raise ValueError("Invalid sampling_freq, must be 'min', 'max', or numeric")
    sf_name = stream_info.loc[stream_info["sf"] == sf, "name"].values
    print(f"Using {sf_type} sampling rate: {sf} Hz ({sf_name})")

    max_first_ts = stream_info["first_ts"].max()
    max_first_ts_name = stream_info.loc[
        stream_info["first_ts"] == max_first_ts, "name"
    ].values
    print(f"Using latest start timestamp: {max_first_ts} ({max_first_ts_name})")

    min_last_ts = stream_info["last_ts"].min()
    min_last_ts_name = stream_info.loc[
        stream_info["last_ts"] == min_last_ts, "name"
    ].values
    print(f"Using earliest last timestamp: {min_last_ts} ({min_last_ts_name})")

    new_ts = np.arange(
        max_first_ts,
        min_last_ts,
        int(1e9 / sf),
        dtype=np.int64,
    )

    concat_data = pd.DataFrame(index=new_ts)
    for stream in stream_info["stream"]:
        interp_data = stream.interpolate(
            new_ts, interp_float_kind, interp_other_kind, inplace=inplace
        ).data
        assert concat_data.shape[0] == interp_data.shape[0]
        assert concat_data.index.equals(interp_data.index)
        concat_data = pd.concat([concat_data, interp_data], axis=1)
        assert concat_data.shape[0] == interp_data.shape[0]
    concat_data.index.name = "timestamp [ns]"
    return concat_data


VALID_EVENTS = {
    "blink",
    "blinks",
    "fixation",
    "fixations",
    "saccade",
    "saccades",
    "event",
    "events",
}


def concat_events(
    rec: "Recording",
    event_names: str | list[str],
) -> pd.DataFrame:
    """
    Concatenate different events. All columns in the selected event type will be
    present in the final DataFrame. An additional ``type`` column denotes the event
    type. If ``"events"`` is in ``event_names``, its ``timestamp [ns]`` column will be
    renamed to ``start timestamp [ns]``, and the ``name`` and ``type`` columns will
    be renamed to ``message name`` and ``message type`` respectively to prevent confusion
    between physiological events and user-supplied messages.

    Parameters
    ----------
    rec : Recording
        Recording object containing the events to concatenate.
    event_names : list of str
        List of event names to concatenate. Event names must be in
        ``{"blinks", "fixations", "saccades", "events"}``
        (singular forms are tolerated).

    Returns
    -------
    pandas.DataFrame
        Concatenated events.
    """
    if isinstance(event_names, str):
        if event_names == "all":
            event_names = list(VALID_EVENTS)
        else:
            raise ValueError(
                "Invalid event_names, must be 'all' or a list of event names."
            )

    if len(event_names) <= 1:
        raise ValueError("Must provide at least two events to concatenate.")

    event_names = [ev.lower() for ev in event_names]
    # Check if all events are valid
    if not all([ev in VALID_EVENTS for ev in event_names]):
        raise ValueError(f"Invalid event name, can only be {VALID_EVENTS}")

    concat_data = pd.DataFrame(
        {
            "end timestamp [ns]": pd.Series(dtype="Int64"),
            "duration [ms]": pd.Series(dtype="float64"),
            "type": pd.Series(dtype="str"),
        }
    )
    print("Concatenating events:")
    if "blinks" in event_names or "blink" in event_names:
        if rec.blinks is None:
            raise ValueError("Cannot load blink data.")
        data = rec.blinks.data
        data["type"] = "blink"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tBlinks")
    if "fixations" in event_names or "fixation" in event_names:
        if rec.fixations is None:
            raise ValueError("Cannot load fixation data.")
        data = rec.fixations.data
        data["type"] = "fixation"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tFixations")
    if "saccades" in event_names or "saccade" in event_names:
        if rec.saccades is None:
            raise ValueError("Cannot load saccade data.")
        data = rec.saccades.data
        data["type"] = "saccade"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tSaccades")
    if "events" in event_names or "event" in event_names:
        if rec.events is None:
            raise ValueError("Cannot load event data.")
        data = rec.events.data
        data.index.name = "start timestamp [ns]"
        data = data.rename(columns={"name": "message name", "type": "message type"})
        data["type"] = "event"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tEvents")
    concat_data = concat_data.sort_index()
    return concat_data
