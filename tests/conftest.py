import numpy as np
import pandas as pd
import pytest

from pyneon import Events, Recording, Stream


@pytest.fixture(scope="package")
def simple_streams():
    ts1 = np.arange(1e9, 5e9, 1e9 / 50)  # 50 Hz
    ts2 = np.arange(2e9, 6e9, 1e9 / 100)  # 100 Hz
    ts3 = np.arange(3e9, 7e9, 1e9 / 200)  # 500 Hz
    ts4 = np.arange(4e9, 8e9, int(1e9 / 30))  # 30 Hz
    df1 = pd.DataFrame(
        np.random.rand(len(ts1), 2), index=ts1, columns=["gaze x [px]", "gaze y [px]"]
    )
    df2 = pd.DataFrame(
        np.random.rand(len(ts2), 2),
        index=ts2,
        columns=["pupil diameter left [mm]", "pupil diameter right [mm]"],
    )
    df3 = pd.DataFrame(
        np.random.rand(len(ts3), 2),
        index=ts3,
        columns=["gyro x [deg/s]", "gyro y [deg/s]"],
    )
    df4 = pd.DataFrame(
        np.random.rand(len(ts4), 2), index=ts4, columns=["custom x", "custom y"]
    )
    for df in [df1, df2, df3, df4]:
        df.index.name = "timestamp [ns]"
    with pytest.warns(UserWarning, match="Following columns not in known data types"):
        stream_4 = Stream(df4)
    return Stream(df1), Stream(df2), Stream(df3), stream_4
