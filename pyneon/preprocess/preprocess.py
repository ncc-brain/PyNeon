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
                old_data["timestamp [ns]"], old_data[col], kind=float_kind
            )(new_ts)
        else:
            resamp_data[col] = interpolate.interp1d(
                old_data["timestamp [ns]"], old_data[col], kind=other_kind
            )(new_ts)
        resamp_data[col] = resamp_data[col].astype(old_data[col].dtype)
    return resamp_data


_VALID_STREAMS = ["3d_eye_states", "eye_states", "gaze", "imu"]


def concat_streams(
    rec: "NeonRecording",
    stream_names: list[str],
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
    stream_names : list of str
        List of stream names to concatenate. Stream names must be in
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
    if len(stream_names) <= 1:
        raise ValueError("Must provide at least two streams to concatenate.")

    stream_names = [ch.lower() for ch in stream_names]
    # Check if all streams are valid
    if not all([ch in _VALID_STREAMS for ch in stream_names]):
        raise ValueError(f"Invalid stream name, can only be {_VALID_STREAMS}")

    ch_info = pd.DataFrame(columns=["stream", "name", "sf", "first_ts", "last_ts"])
    print("Concatenating streams:")
    if "gaze" in stream_names:
        if rec.gaze is None:
            raise ValueError("Cannnot load gaze data.")
        ch_info = pd.concat(
            [
                ch_info,
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
        ch_info = pd.concat(
            [
                ch_info,
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
        ch_info = pd.concat(
            [
                ch_info,
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
        sf = ch_info["sf"].min()
        sf_type = "lowest"
    elif sampling_freq == "max":
        sf = ch_info["sf"].max()
        sf_type = "highest"
    elif isinstance(sampling_freq, (int, float)):
        sf = sampling_freq
        sf_type = "customized"
    else:
        raise ValueError("Invalid sampling_freq, must be 'min', 'max', or numeric")
    sf_name = ch_info.loc[ch_info["sf"] == sf, "name"].values
    print(f"Using {sf_type} sampling rate: {sf} Hz ({sf_name})")

    max_first_ts = ch_info["first_ts"].max()
    max_first_ts_name = ch_info.loc[ch_info["first_ts"] == max_first_ts, "name"].values
    print(f"Using latest start timestamp: {max_first_ts} ({max_first_ts_name})")

    min_last_ts = ch_info["last_ts"].min()
    min_last_ts_name = ch_info.loc[ch_info["last_ts"] == min_last_ts, "name"].values
    print(f"Using earliest last timestamp: {min_last_ts} ({min_last_ts_name})")

    new_ts = np.arange(
        max_first_ts,
        min_last_ts,
        int(1e9 / sf),
        dtype=np.int64,
    )

    concat_data = pd.DataFrame(data=new_ts, columns=["timestamp [ns]"], dtype="Int64")
    concat_data["time [s]"] = (new_ts - new_ts[0]) / 1e9
    for ch in ch_info["stream"]:
        resamp_df = ch.resample(
            new_ts, resamp_float_kind, resamp_other_kind, inplace=inplace
        )
        assert concat_data.shape[0] == resamp_df.shape[0]
        assert concat_data["timestamp [ns]"].equals(resamp_df["timestamp [ns]"])
        concat_data = pd.merge(
            concat_data, resamp_df, on=["timestamp [ns]", "time [s]"], how="inner"
        )
        assert concat_data.shape[0] == resamp_df.shape[0]
    return concat_data
