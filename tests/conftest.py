import numpy as np
import pandas as pd
import pytest

from pyneon import Events, Recording, Stream
from pyneon.utils.variables import nominal_sampling_rates


@pytest.fixture(scope="package")
def simple_streams():
    gaze_ts = np.arange(1e9, 5e9, 1e9 / 50)  # 50 Hz
    eye_states_ts = np.arange(2e9, 6e9, 1e9 / 100)  # 100 Hz
    imu_ts = np.arange(3e9, 7e9, 1e9 / 200)  # 200 Hz
    custom_ts = np.arange(4e9, 8e9, int(1e9 / 30))  # 30 Hz
    gaze_df = pd.DataFrame(
        np.random.rand(len(gaze_ts), 2),
        index=gaze_ts,
        columns=["gaze x [px]", "gaze y [px]"],
    )
    eye_states_df = pd.DataFrame(
        np.random.rand(len(eye_states_ts), 2),
        index=eye_states_ts,
        columns=["pupil diameter left [mm]", "pupil diameter right [mm]"],
    )
    imu_df = pd.DataFrame(
        np.random.rand(len(imu_ts), 2),
        index=imu_ts,
        columns=["gyro x [deg/s]", "gyro y [deg/s]"],
    )
    custom_df = pd.DataFrame(
        np.random.rand(len(custom_ts), 2),
        index=custom_ts,
        columns=["custom x", "custom y"],
    )
    for df in [gaze_df, eye_states_df, imu_df, custom_df]:
        df.index.name = "timestamp [ns]"

    gaze = Stream(gaze_df)
    assert gaze.type == "gaze"
    assert gaze.sampling_freq_nominal == nominal_sampling_rates["gaze"]

    eye_states = Stream(eye_states_df)
    assert eye_states.type == "eye_states"
    assert eye_states.sampling_freq_nominal == nominal_sampling_rates["eye_states"]

    imu = Stream(imu_df)
    assert imu.type == "imu"
    assert imu.sampling_freq_nominal == nominal_sampling_rates["imu"]

    with pytest.warns(UserWarning, match="Following columns not in known data types"):
        custom = Stream(custom_df)
    assert custom.type == "custom"
    assert custom.sampling_freq_nominal is None

    return Stream(gaze_df), Stream(eye_states_df), Stream(imu_df), custom
