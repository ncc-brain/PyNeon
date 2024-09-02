import pandas as pd
import numpy as np

from typing import TYPE_CHECKING
from scipy import interpolate

if TYPE_CHECKING:
    from ..recording import NeonRecording


def resample(
    new_ts: np.ndarray,
    old_df: pd.DataFrame,
    float_kind: str = "linear",
    other_kind: str = "nearest",
) -> pd.DataFrame:
    # Check that 'timestamp [ns]' is in the columns
    if "timestamp [ns]" not in old_df.columns:
        raise ValueError("old_df must contain a 'timestamp [ns]' column")
    # Check that new_ts is monotonicically increasing
    if np.any(np.diff(new_ts) < 0):
        raise ValueError("new_ts must be monotonically increasing")
    # Create a new dataframe with the new timestamps
    resamp_data = pd.DataFrame(data=new_ts, columns=["timestamp [ns]"], dtype="Int64")
    resamp_data["time [s]"] = (new_ts - new_ts[0]) / 1e9
    for col in old_df.columns:
        if col == "timestamp [ns]" or col == "time [s]":
            continue
        if pd.api.types.is_float_dtype(old_df[col]):
            resamp_data[col] = interpolate.interp1d(
                old_df["timestamp [ns]"], old_df[col], kind=float_kind
            )(new_ts)
        else:
            resamp_data[col] = interpolate.interp1d(
                old_df["timestamp [ns]"], old_df[col], kind=other_kind
            )(new_ts)
        resamp_data[col] = resamp_data[col].astype(old_df[col].dtype)
    return resamp_data


_VALID_CHANNELS = ["3d_eye_states", "eye_states", "gaze", "imu"]


def concat_channels(
    rec: "NeonRecording",
    ch_names: list[str],
    downsample: bool = True,
    resamp_float_kind: str = "linear",
    resamp_other_kind: str = "nearest",
) -> pd.DataFrame:
    """Combine multiple channels into a single dataframe.

    Resampling is necessary to align all signals to the same timestamps.
    If channels have different sampling rates, the lowest sampling rate is used.
    """
    if len(ch_names) <= 1:
        raise ValueError("Must provide at least two channels to concatenate.")

    ch_names = [ch.lower() for ch in ch_names]
    # Check if all channels are valid
    if not all([ch in _VALID_CHANNELS for ch in ch_names]):
        raise ValueError(f"Invalid channel name, can only be {_VALID_CHANNELS}")

    ch_info = pd.DataFrame(columns=["signal", "name", "sf", "first_ts", "last_ts"])
    print("Concatenating channels:")
    if "gaze" in ch_names:
        ch_info = pd.concat(
            [
                ch_info,
                pd.Series(
                    {
                        "signal": rec.gaze,
                        "name": "gaze",
                        "sf": rec.gaze.sampling_rate_nominal,
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
    if "3d_eye_states" in ch_names or "eye_states" in ch_names:
        ch_info = pd.concat(
            [
                ch_info,
                pd.Series(
                    {
                        "signal": rec.eye_states,
                        "name": "3d_eye_states",
                        "sf": rec.eye_states.sampling_rate_nominal,
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
    if "imu" in ch_names:
        ch_info = pd.concat(
            [
                ch_info,
                pd.Series(
                    {
                        "signal": rec.imu,
                        "name": "imu",
                        "sf": rec.imu.sampling_rate_nominal,
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
    if downsample:
        sf = ch_info["sf"].min()
        sf_type = 'Lowest'
    else:
        sf = ch_info["sf"].max()
        sf_type = 'Highest'
    sf_name = ch_info.loc[ch_info["sf"] == sf, "name"].values
    print(f"{sf_type} sampling rate: {sf} Hz ({sf_name})")

    max_first_ts = ch_info["first_ts"].max()
    max_first_ts_name = ch_info.loc[ch_info["first_ts"] == max_first_ts, "name"].values
    print(f"Latest start timestamp: {max_first_ts} ({max_first_ts_name})")

    min_last_ts = ch_info["last_ts"].min()
    min_last_ts_name = ch_info.loc[ch_info["last_ts"] == min_last_ts, "name"].values
    print(f"Earliest last timestamp: {min_last_ts} ({min_last_ts_name})")

    new_ts = np.arange(
        max_first_ts,
        min_last_ts,
        int(1e9 / sf),
        dtype=np.int64,
    )

    concat_data = pd.DataFrame(data=new_ts, columns=["timestamp [ns]"], dtype="Int64")
    concat_data["time [s]"] = (new_ts - new_ts[0]) / 1e9
    for ch in ch_info["signal"]:
        resamp_df = resample(new_ts, ch.data, resamp_float_kind, resamp_other_kind)
        assert concat_data.shape[0] == resamp_df.shape[0]
        assert concat_data["timestamp [ns]"].equals(resamp_df["timestamp [ns]"])
        concat_data = pd.merge(
            concat_data, resamp_df, on=["timestamp [ns]", "time [s]"], how="inner"
        )
        assert concat_data.shape[0] == resamp_df.shape[0]
    return concat_data
