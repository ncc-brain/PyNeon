import pandas as pd
import numpy as np
from tqdm import tqdm

from numbers import Number
from typing import TYPE_CHECKING, Optional
from scipy.interpolate import interp1d

from ..utils import _check_stream_data

if TYPE_CHECKING:
    from ..recording import NeonRecording


def interpolate(
    new_ts: np.ndarray,
    data: pd.DataFrame,
    float_kind: str = "cubic",
    other_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Interpolate a data stream to a new set of timestamps.

    Parameters
    ----------
    new_ts : np.ndarray
        An array of new timestamps (in nanoseconds)
        at which to evaluate the interpolant.
    data : pd.DataFrame
        Data to interpolate. Must have a monotonically increasing
        index named ``timestamp [ns]``.
    float_kind : str, optional
        Kind of interpolation applied on columns of float type,
        by default ``"cubic"``. For details see :class:`scipy.interpolate.interp1d`.
    other_kind : str, optional
        Kind of interpolation applied on columns of other types,
        by default ``"nearest"``. For details see :class:`scipy.interpolate.interp1d`.

    Returns
    -------
    pandas.DataFrame
        Interpolated data.
    """
    _check_stream_data(data)
    new_ts = np.sort(new_ts).astype(np.int64)
    new_data = pd.DataFrame(index=new_ts, columns=data.columns)
    for col in data.columns:
        # Float columns are interpolated with float_kind
        if pd.api.types.is_float_dtype(data[col]):
            new_data[col] = interp1d(
                data.index,
                data[col],
                kind=float_kind,
                bounds_error=False,
            )(new_ts)
        # Other columns are interpolated with other_kind
        else:
            new_data[col] = interp1d(
                data.index,
                data[col],
                kind=other_kind,
                bounds_error=False,
            )(new_ts)
        # Ensure the new column has the same dtype as the original
        new_data[col] = new_data[col].astype(data[col].dtype)
    return new_data


def window_average(
    new_ts: np.ndarray,
    data: pd.DataFrame,
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Take the average over a time window to obtain smoothed data at new timestamps.

    Parameters
    ----------
    new_ts : np.ndarray
        An array of new timestamps (in nanoseconds)
        at which to compute the windowed averages.
        The median interval between these new timestamps must be larger than
        the median interval between the original data timestamps, i.e.,
        ``np.median(np.diff(new_ts)) > np.median(np.diff(data.index))``.
        In other words, only downsampling is supported.
    data : pd.DataFrame
        Data to apply window average to. Must have a monotonically increasing
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
    pd.DataFrame
        Data with window average applied.
    """
    _check_stream_data(data)
    new_ts = np.sort(new_ts).astype(np.int64)
    new_ts_median_diff = np.median(np.diff(new_ts))
    original_ts_median_diff = np.median(np.diff(data.index))

    # Check for downsampling
    if new_ts_median_diff < original_ts_median_diff:
        raise ValueError(
            "new_ts must have a lower sampling frequency than the old data"
        )

    if window_size is None:
        window_size = int(new_ts_median_diff)

    if window_size < original_ts_median_diff:
        raise ValueError(
            "Window size must be larger than the median difference "
            "between the old timestamps, otherwise data could be interleaved by NAs."
        )

    new_data = pd.DataFrame(index=new_ts, columns=data.columns).astype(data.dtypes)
    non_float_cols = data.select_dtypes(exclude="float").columns

    for ts in tqdm(new_ts, desc="Computing window averages"):
        lower_bound = ts - window_size / 2
        upper_bound = ts + window_size / 2

        # Select data within the window
        window_data = data[
            (data.index >= lower_bound) & (data.index <= upper_bound)
        ].copy()
        if not window_data.empty:
            # Take the mean of the window data
            mean_values = window_data.mean(axis=0)
            # Round non-float columns so that they can be safely set to the original data type
            for non_float_col in non_float_cols:
                if not pd.isna(col_value := mean_values[non_float_col]):
                    mean_values[non_float_col] = round(col_value)
            new_data.loc[ts, :] = mean_values
        else:
            new_data.loc[ts, :] = pd.NA

    return new_data


_VALID_STREAMS = {"3d_eye_states", "eye_states", "gaze", "imu"}


def concat_streams(
    rec: "NeonRecording",
    stream_names: str | list[str] = "all",
    sampling_freq: Number | str = "min",
    interp_float_kind: str = "cubic",
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
    rec : :class:`NeonRecording`
        NeonRecording object containing the streams to concatenate.
    stream_names : str or list of str
        Stream names to concatenate. If "all", then all streams will be used.
        If a list, items must be in ``{"gaze", "imu", "eye_states"}``
        (``"3d_eye_states"``) is also tolerated as an alias for ``"eye_states"``).
    sampling_freq : float or int or str, optional
        Sampling frequency of the concatenated streams.
        If numeric, the streams will be interpolated to this frequency.
        If ``"min"`` (default), the lowest nominal sampling frequency
        of the selected streams will be used.
        If ``"max"``, the highest nominal sampling frequency will be used.
    interp_float_kind : str, optional
        Kind of interpolation applied on columns of float type,
        Defaults to ``"cubic"``. For details see :class:`scipy.interpolate.interp1d`.
    interp_other_kind : str, optional
        Kind of interpolation applied on columns of other types.
        Defaults to ``"nearest"``.
    inplace : bool, optional
        Replace selected stream data with interpolated data during concatenation
        if``True``. Defaults to ``False``.

    Returns
    -------
    concat_data : :class:`pandas.DataFrame`
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
    if "gaze" in stream_names:
        if rec.gaze is None:
            raise ValueError("Cannnot load gaze data.")
        stream_info = pd.concat(
            [
                stream_info,
                pd.Series(
                    {
                        "stream": rec.gaze,
                        "name": "gaze",
                        "sf": rec.gaze.sampling_freq_nominal,
                        "first_ts": rec.gaze.first_ts,
                        "last_ts": rec.gaze.last_ts,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )
        print("\tGaze")
    if "3d_eye_states" in stream_names or "eye_states" in stream_names:
        if rec.eye_states is None:
            raise ValueError("Cannnot load eye states data.")
        stream_info = pd.concat(
            [
                stream_info,
                pd.Series(
                    {
                        "stream": rec.eye_states,
                        "name": "3d_eye_states",
                        "sf": rec.eye_states.sampling_freq_nominal,
                        "first_ts": rec.eye_states.first_ts,
                        "last_ts": rec.eye_states.last_ts,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )
        print("\t3D eye states")
    if "imu" in stream_names:
        if rec.imu is None:
            raise ValueError("Cannnot load IMU data.")
        stream_info = pd.concat(
            [
                stream_info,
                pd.Series(
                    {
                        "stream": rec.imu,
                        "name": "imu",
                        "sf": rec.imu.sampling_freq_nominal,
                        "first_ts": rec.imu.first_ts,
                        "last_ts": rec.imu.last_ts,
                    }
                )
                .to_frame()
                .T,
            ],
            ignore_index=True,
        )
        print("\tIMU")

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
    rec: "NeonRecording",
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
    rec : :class:`NeonRecording`
        NeonRecording object containing the events to concatenate.
    event_names : list of str
        List of event names to concatenate. Event names must be in
        ``{"blinks", "fixations", "saccades", "events"}``
        (singular forms are tolerated).

    Returns
    -------
    concat_events : :class:`pandas.DataFrame`
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
            raise ValueError("Cannnot load blink data.")
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
            raise ValueError("Cannnot load fixation data.")
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
            raise ValueError("Cannnot load saccade data.")
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
            raise ValueError("Cannnot load event data.")
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
