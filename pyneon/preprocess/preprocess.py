import pandas as pd
import numpy as np

from typing import TYPE_CHECKING, Union
from scipy import interpolate

if TYPE_CHECKING:
    from ..recording import NeonRecording


def resample(
    new_ts: np.ndarray,
    old_data: pd.DataFrame,
    float_kind: str = "linear",
    other_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Resample the stream to a new set of timestamps.

    Parameters
    ----------
    new_ts : np.ndarray, optional
        New timestamps to resample the stream to. If ``None``,
        the stream is resampled to its nominal sampling frequency according to
        https://pupil-labs.com/products/neon/specs.
    old_data : pd.DataFrame
        Data to resample. Must contain a monotonically increasing
        ``timestamp [ns]`` column.
    float_kind : str, optional
        Kind of interpolation applied on columns of float type,
        by default "linear". For details see :class:`scipy.interpolate.interp1d`.
    other_kind : str, optional
        Kind of interpolation applied on columns of other types,
        by default "nearest".

    Returns
    -------
    pandas.DataFrame
        Resampled data.
    """
    # Check that 'timestamp [ns]' is in the columns
    if "timestamp [ns]" not in old_data.columns:
        raise ValueError("old_data must contain a 'timestamp [ns]' column")
    # Check that new_ts is monotonicically increasing
    if np.any(np.diff(new_ts) < 0):
        raise ValueError("new_ts must be monotonically increasing")
    # Create a new dataframe with the new timestamps
    resamp_data = pd.DataFrame(data=new_ts, columns=["timestamp [ns]"], dtype="Int64")
    resamp_data["time [s]"] = (new_ts - new_ts[0]) / 1e9

    for col in old_data.columns:
        if col == "timestamp [ns]" or col == "time [s]":
            continue
        if pd.api.types.is_float_dtype(old_data[col]):
            resamp_data[col] = interpolate.interp1d(
                old_data["timestamp [ns]"],
                old_data[col],
                kind=float_kind,
                bounds_error=False,
            )(new_ts)
        else:
            resamp_data[col] = interpolate.interp1d(
                old_data["timestamp [ns]"],
                old_data[col],
                kind=other_kind,
                bounds_error=False,
            )(new_ts)
        resamp_data[col] = resamp_data[col].astype(old_data[col].dtype)
    return resamp_data


def rolling_average(
    new_ts: np.ndarray,
    old_data: pd.DataFrame,
    time_column: str = "timestamp [ns]",
) -> pd.DataFrame:
    """
    Apply rolling average over a time window to resampled data.

    Parameters
    ----------
    new_ts : np.ndarray
        New timestamps to resample the stream to.
    old_data : pd.DataFrame
        Data to apply rolling average to.
    time_column : str, optional
        Name of the time column in the data, by default "timestamp [ns]".

    Returns
    -------
    pd.DataFrame
        Data with rolling averages applied.
    """
    # Assert that 'timestamp [ns]' is present and monotonic
    if "timestamp [ns]" not in old_data.columns:
        raise ValueError("old_data must contain a 'timestamp [ns]' column")

    if not np.all(np.diff(old_data["timestamp [ns]"]) > 0):
        # call resample function to ensure monotonicity
        old_data = resample(None, old_data)

    # assert that the new_ts has a lower sampling frequency than the old data
    if np.mean(np.diff(new_ts)) < np.mean(np.diff(old_data[time_column])):
        raise ValueError(
            "new_ts must have a lower sampling frequency than the old data"
        )

    # Create a new DataFrame for the downsampled data
    downsampled_data = pd.DataFrame(data=new_ts, columns=[time_column], dtype="Int64")
    downsampled_data["time [s]"] = (new_ts - new_ts[0]) / 1e9

    # Convert window_size to nanoseconds
    window_size = np.mean(np.diff(new_ts))

    # Loop through each column (excluding time columns) to compute the downsampling
    for col in old_data.columns:
        if col == time_column or col == "time [s]":
            continue

        # Initialize an empty list to store the downsampled values
        downsampled_values = []

        # Loop through each new timestamp
        for ts in new_ts:
            # Define the time window around the current new timestamp
            lower_bound = ts - window_size / 2
            upper_bound = ts + window_size / 2

            # Select rows from old_data that fall within the time window
            window_data = old_data[
                (old_data[time_column] >= lower_bound)
                & (old_data[time_column] <= upper_bound)
            ]

            # Compute the average of the data within this window
            if not window_data.empty:
                mean_value = window_data[col].mean()
            else:
                mean_value = np.nan

            # Append the averaged value to the list
            downsampled_values.append(mean_value)

        # Assign the downsampled values to the new DataFrame
        downsampled_data[col] = downsampled_values

    return downsampled_data


_VALID_STREAMS = {"3d_eye_states", "eye_states", "gaze", "imu"}


def concat_streams(
    rec: "NeonRecording",
    stream_names: Union[str, list[str]] = "all",
    sampling_freq: Union[float, int, str] = "min",
    resamp_float_kind: str = "linear",
    resamp_other_kind: str = "nearest",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Concatenate data from different streams under common timestamps.
    Since the streams may have different timestamps and sampling frequencies,
    resampling of all streams to a set of common timestamps is performed.
    The latest start timestamp and earliest last timestamp of the selected streams
    are used to define the common timestamps.

    Parameters
    ----------
    rec : :class:`NeonRecording`
        NeonRecording object containing the streams to concatenate.
    stream_names : str or list of str
        Stream names to concatenate. If "all", then all streams will be used.
        If a list, items must be in
        ``{"gaze", "imu", "eye_states", "3d_eye_states"}``.
    sampling_freq : float or int or str, optional
        Sampling frequency to resample the streams to.
        If numeric, the streams will be resampled to this frequency.
        If ``"min"``, the lowest nominal sampling frequency
        of the selected streams will be used.
        If ``"max"``, the highest nominal sampling frequency will be used.
        Defaults to ``"min"``.
    resamp_float_kind : str, optional
        Kind of interpolation applied on columns of float type,
        Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
    resamp_other_kind : str, optional
        Kind of interpolation applied on columns of other types.
        Defaults to ``"nearest"``.
    inplace : bool, optional
        Replace selected stream data with resampled data during concatenation
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

    concat_data = pd.DataFrame(data=new_ts, columns=["timestamp [ns]"], dtype="Int64")
    concat_data["time [s]"] = (new_ts - new_ts[0]) / 1e9
    for stream in stream_info["stream"]:
        resamp_df = stream.resample(
            new_ts, resamp_float_kind, resamp_other_kind, inplace=inplace
        )
        assert concat_data.shape[0] == resamp_df.shape[0]
        assert concat_data["timestamp [ns]"].equals(resamp_df["timestamp [ns]"])
        concat_data = pd.merge(
            concat_data, resamp_df, on=["timestamp [ns]", "time [s]"], how="inner"
        )
        assert concat_data.shape[0] == resamp_df.shape[0]
    return concat_data


VALID_EVENTS = {"blinks", "fixations", "saccades", "events"}


def concat_events(
    rec: "NeonRecording",
    event_names: Union[str, list[str]],
) -> pd.DataFrame:
    """
    Concatenate events from different streams under common timestamps.
    The latest start timestamp and earliest last timestamp of the selected events
    are used to define the common timestamps.

    Parameters
    ----------
    rec : :class:`NeonRecording`
        NeonRecording object containing the events to concatenate.
    event_names : list of str
        List of event names to concatenate. Event names must be in
        ``{"blinks", "fixations", "saccades", "events"}``.

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
