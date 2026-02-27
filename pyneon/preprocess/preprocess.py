from numbers import Number
from typing import TYPE_CHECKING, Literal, Optional
from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from scipy.interpolate import interp1d

from ..utils import _validate_neon_tabular_data, _validate_df_columns
from ..utils.doc_decorators import fill_doc

if TYPE_CHECKING:
    from ..events import Events
    from ..recording import Recording


@fill_doc
def interpolate(
    new_ts: np.ndarray,
    data: pd.DataFrame,
    float_kind: str | int = "linear",
    other_kind: str | int = "nearest",
    max_gap_ms: Optional[Number] = 500,
) -> pd.DataFrame:
    """
    Interpolate a data stream to a new set of timestamps.

    Data columns of float type are interpolated using ``float_kind``,
    while other columns use ``other_kind``. This distinction allows
    for appropriate interpolation methods based on data type.

    Parameters
    ----------
    new_ts : numpy.ndarray
        An array of new timestamps (in nanoseconds) at which to evaluate
        the interpolant.
    data : pandas.DataFrame
        Source data to interpolate. Must have a monotonically increasing
        index named ``timestamp [ns]``.

    %(interp_kind_params)s
    %(max_gap_ms_param)s

    Returns
    -------
    pandas.DataFrame
        Interpolated data with the same columns and dtypes as ``data``
        and indexed by ``new_ts``.

    Notes
    -----
    - If ``new_ts`` contains timestamps outside the range of ``data.index``,
      the corresponding rows will contain NaN.
    """
    _validate_neon_tabular_data(data)
    new_ts = np.sort(new_ts).astype("int64")

    # Track which timestamps are invalid
    invalid_mask = np.zeros(len(new_ts), dtype=bool)

    # First, mark timestamps outside the data range
    out_of_range = (new_ts < data.index[0]) | (new_ts > data.index[-1])
    invalid_mask |= out_of_range
    n_out_of_bounds = np.sum(out_of_range)
    if n_out_of_bounds > 0:
        warn(
            f"{n_out_of_bounds} out of {len(new_ts)} requested timestamps are outside "
            f"the data time range and will have empty data.",
            UserWarning,
        )

    # Then, for in-range timestamps, check max_gap_ms constraint
    if max_gap_ms is not None:
        max_gap_ns = int(max_gap_ms * 1e6)
        old_ts = data.index.to_numpy()

        # Only check timestamps that are within the data range
        in_range_mask = ~out_of_range
        if np.any(in_range_mask):
            new_ts_in_range = new_ts[in_range_mask]
            idx = np.searchsorted(old_ts, new_ts_in_range, side="left")

            # Check for exact matches first
            exact_match = np.isin(new_ts_in_range, old_ts)

            # For non-exact matches, compute distances to neighbors
            left_dist = np.where(
                idx == 0,
                np.inf,
                new_ts_in_range - old_ts[np.clip(idx - 1, 0, len(old_ts) - 1)],
            )
            right_dist = np.where(
                idx == len(old_ts),
                np.inf,
                old_ts[np.clip(idx, 0, len(old_ts) - 1)] - new_ts_in_range,
            )

            # Valid if exact match OR both neighbors are close enough
            valid_in_range = exact_match | (
                (left_dist < max_gap_ns) & (right_dist < max_gap_ns)
            )

            # Mark invalid timestamps
            invalid_in_range = ~valid_in_range
            invalid_mask[in_range_mask] |= invalid_in_range

            n_gap_violations = np.sum(invalid_in_range)
            if n_gap_violations > 0:
                warn(
                    f"{n_gap_violations} out of {len(new_ts)} requested timestamps exceed "
                    f"max_gap_ms={max_gap_ms} relative to neighboring samples and will have empty data.",
                    UserWarning,
                )

    if other_kind not in ("nearest", "nearest-up", "previous", "next"):
        warn(
            f"Interpolation kind '{other_kind}' for non-float columns "
            "is not among the recommended kinds ('nearest', 'nearest-up', "
            "'previous', 'next'). Numerical interpolation could result in "
            "invalid values.",
            UserWarning,
        )

    new_data = pd.DataFrame(index=new_ts, columns=data.columns)
    new_data.index.name = data.index.name

    for col in data.columns:
        s = data[col]

        if is_float_dtype(s):  # Interp with float_kind
            vals = interp1d(s.index, s, kind=float_kind, bounds_error=False)(new_ts)
            new_data[col] = vals.astype(s.dtype, copy=False)
        else:  # Interp with other_kind
            vals = interp1d(s.index, s, kind=other_kind, bounds_error=False)(new_ts)
            try:
                new_data[col] = vals.astype(s.dtype, copy=False)
            except (TypeError, ValueError):  # fallback in case .astype fails
                new_data[col] = vals

    # Set data to NaN for timestamps that are out of range
    if np.any(invalid_mask):
        new_data.loc[invalid_mask] = np.nan

    return new_data


@fill_doc
def interpolate_events(
    data: pd.DataFrame,
    events: "Events",
    buffer: Number | tuple[Number, Number] = 0.05,
    float_kind: str | int = "linear",
    other_kind: str | int = "nearest",
    max_gap_ms: Optional[Number] = 500,
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
        Events instance containing the events to interpolate.
        The events must have ``start timestamp [ns]`` and
        ``end timestamp [ns]`` columns.
    buffer : numbers.Number or , optional
        The time before and after an event (in seconds) to consider invalid.
        If a single number is provided, the same buffer is applied
        to both before and after the event.
        Defaults to 0.05.

    %(interp_kind_params)s
    %(max_gap_ms_param)s

    Returns
    -------
    pandas.DataFrame
        Interpolated data with the same columns and dtypes as ``data``
        and indexed by ``data.index``.
    """
    _validate_neon_tabular_data(data)

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
        max_gap_ms=max_gap_ms,
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

    _validate_neon_tabular_data(data)
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


def compute_azimuth_and_elevation(
    data: pd.DataFrame,
    method: Literal["linear"] = "linear",
    overwrite: bool = False,
) -> None:
    """
    Append gaze azimuth and elevation angles (in degrees) to gaze data
    based on gaze pixel coordinates. Operates in-place.

    Parameters
    ----------
    data : pandas.DataFrame
        Gaze data containing ``gaze x [px]`` and ``gaze y [px]`` columns.
    method : str, optional
        Method to compute gaze angles. Currently only "linear" is supported.
        Defaults to "linear".

    Returns
    -------
    None
        The function modifies the input DataFrame in-place by adding two new columns:
        ``azimuth [deg]`` and ``elevation [deg]``.
    """
    if not overwrite and (
        "azimuth [deg]" in data.columns or "elevation [deg]" in data.columns
    ):
        raise ValueError(
            "Stream data already contains azimuth and/or elevation columns. "
            "Use overwrite=True to overwrite existing columns."
        )

    required_cols = ["gaze x [px]", "gaze y [px]"]
    camera_resolution = [1600, 1200]  # Pupil Neon camera resolution in pixels
    camera_fov = [103, 77]  # Pupil Neon camera field of view in degrees
    _validate_df_columns(data, required_cols, df_name="gaze data")
    if method == "linear":
        data["azimuth [deg]"] = (
            (data["gaze x [px]"] - camera_resolution[0] / 2)
            / camera_resolution[0]
            * camera_fov[0]
        )
        data["elevation [deg]"] = (
            -(data["gaze y [px]"] - camera_resolution[1] / 2)
            / camera_resolution[1]
            * camera_fov[1]
        )
    else:
        raise NotImplementedError(f"Method '{method}' not implemented for gaze angles.")


_VALID_STREAMS = {"3d_eye_states", "eye_states", "gaze", "imu"}


@fill_doc
def concat_streams(
    rec: "Recording",
    stream_names: str | list[str] = "all",
    sampling_freq: int | float | str = "min",
    float_kind: str | int = "linear",
    other_kind: str | int = "nearest",
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
        Recording instance containing the streams to concatenate.
    stream_names : str or list of str
        Stream names to concatenate. If "all" (default), then all streams will be used.
        If a list, items must be in ``{"gaze", "imu", "eye_states"}``
        ("3d_eye_states") is also tolerated as an alias for "eye_states").
    sampling_freq : int, float, or str, optional
        Sampling frequency of the concatenated streams.
        If numeric, all streams will be interpolated to this frequency.
        If "min" (default), the lowest nominal sampling frequency
        of the selected streams will be used.
        If "max", the highest nominal sampling frequency will be used.

    %(interp_kind_params)s

    %(inplace_param)s

    Returns
    -------
    pandas.DataFrame
        Concatenated data.
    """
    if isinstance(stream_names, str):  # Only "all" is allowed as a string
        if stream_names == "all":
            stream_names = list(_VALID_STREAMS)
        else:
            raise ValueError(
                "Invalid stream_names, must be 'all' or a list of stream names."
            )

    # Normalize stream names and handle aliases
    stream_names = [
        ("eye_states" if ch.lower() == "3d_eye_states" else ch.lower())
        for ch in stream_names
    ]
    stream_names = list(
        dict.fromkeys(stream_names)
    )  # Remove duplicates while preserving order

    # Check if all streams are valid
    if not all([ch in _VALID_STREAMS for ch in stream_names]):
        raise ValueError(f"Invalid stream name, can only be one of {_VALID_STREAMS}")
    # Check at least two streams are provided
    if len(stream_names) <= 1:
        raise ValueError("Must provide at least two different streams to concatenate.")

    concat_list = []
    print("Concatenating streams:")
    for name in stream_names:
        stream_obj = getattr(rec, name)
        concat_list.append(
            {
                "stream": stream_obj,
                "name": name,
                "sf": stream_obj.sampling_freq_nominal,
                "first_ts": stream_obj.first_ts,
                "last_ts": stream_obj.last_ts,
            }
        )
        print(f"\t{name}")

    streams_info = pd.DataFrame(concat_list)

    # Lowest sampling rate
    if sampling_freq == "min":
        sf = streams_info["sf"].min()
        sf_note = "lowest"
    elif sampling_freq == "max":
        sf = streams_info["sf"].max()
        sf_note = "highest"
    elif isinstance(sampling_freq, (int, float)):
        sf = sampling_freq
        sf_note = "custom"
    else:
        raise ValueError("Invalid sampling_freq, must be 'min', 'max', or numeric")
    sf_name = streams_info.loc[streams_info["sf"] == sf, "name"].values
    print(f"Using {sf_note} sampling rate: {sf} Hz ({sf_name})")

    max_first_ts = streams_info["first_ts"].max()
    max_first_ts_name = streams_info.loc[
        streams_info["first_ts"] == max_first_ts, "name"
    ].values
    print(f"Using latest start timestamp: {max_first_ts} ({max_first_ts_name})")

    min_last_ts = streams_info["last_ts"].min()
    min_last_ts_name = streams_info.loc[
        streams_info["last_ts"] == min_last_ts, "name"
    ].values
    print(f"Using earliest last timestamp: {min_last_ts} ({min_last_ts_name})")

    # Generate new common timestamps
    new_ts = np.arange(
        max_first_ts,
        min_last_ts,
        int(1e9 / sf),
        dtype=np.int64,
    )

    concat_data = pd.DataFrame(index=new_ts)
    for stream in streams_info["stream"]:
        interp_data = stream.interpolate(
            new_ts, float_kind, other_kind, max_gap_ms=None, inplace=inplace
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
    events_names: str | list[str],
) -> pd.DataFrame:
    """
    Concatenate different events. All columns in the selected event type will be
    present in the final DataFrame. An additional ``type`` column denotes the event
    type. If "events" is in ``events_names``, its ``timestamp [ns]`` column will be
    renamed to ``start timestamp [ns]``, and the ``name`` and ``type`` columns will
    be renamed to ``message name`` and ``message type`` respectively to prevent confusion
    between physiological events and user-supplied messages.

    Parameters
    ----------
    rec : Recording
        Recording instance containing the events to concatenate.
    events_names : list of str
        List of event names to concatenate. Event names must be in
        ``{"blinks", "fixations", "saccades", "events"}``
        (singular forms are tolerated).

    Returns
    -------
    pandas.DataFrame
        Concatenated events.
    """
    if isinstance(events_names, str):
        if events_names == "all":
            events_names = list(VALID_EVENTS)
        else:
            raise ValueError(
                "Invalid events_names, must be 'all' or a list of event names."
            )

    if len(events_names) <= 1:
        raise ValueError("Must provide at least two events to concatenate.")

    events_names = [ev.lower() for ev in events_names]
    # Check if all events are valid
    if not all([ev in VALID_EVENTS for ev in events_names]):
        raise ValueError(f"Invalid event name, can only be one of {VALID_EVENTS}")

    concat_data = pd.DataFrame(
        {
            "end timestamp [ns]": pd.Series(dtype="Int64"),
            "duration [ms]": pd.Series(dtype="float64"),
            "type": pd.Series(dtype="str"),
        }
    )
    print("Concatenating events:")
    if "blinks" in events_names or "blink" in events_names:
        data = rec.blinks.data
        data["type"] = "blink"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tBlinks")
    if "fixations" in events_names or "fixation" in events_names:
        data = rec.fixations.data
        data["type"] = "fixation"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tFixations")
    if "saccades" in events_names or "saccade" in events_names:
        data = rec.saccades.data
        data["type"] = "saccade"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tSaccades")
    if "events" in events_names or "event" in events_names:
        data = rec.events.data
        data = data.rename(
            columns={
                "timestamp [ns]": "start timestamp [ns]",
                "name": "message name",
                "type": "message type",
            }
        )
        data["type"] = "event"
        concat_data = (
            data
            if concat_data.empty
            else pd.concat([concat_data, data], ignore_index=False)
        )
        print("\tEvents")
    concat_data = concat_data.sort_index()
    return concat_data
