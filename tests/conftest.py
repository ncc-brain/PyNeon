import numpy as np
import pandas as pd
import pytest

from pyneon import Events, Recording, Stream
from pyneon.utils.variables import nominal_sampling_rates


@pytest.fixture(scope="package")
def simple_streams():
    gaze_ts = np.arange(1e9, 5e9, 1e9 / 50)  # 50 Hz
    eye_states_ts = np.arange(2e9, 6e9, 1e9 / 100)  # 100 Hz
    imu_ts = np.arange(1e9, 4e9, 1e9 / 200)  # 200 Hz
    custom_ts = np.arange(0, 6e9, int(1e9 / 30))  # 30 Hz
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

    return gaze, eye_states, imu, custom


@pytest.fixture(scope="package")
def simple_events():
    blinks_start_ts = np.array([1e9, 2e9, 6e9, 10e9])
    blinks_end_ts = blinks_start_ts + 100e6  # 100 ms blinks
    blinks_end_ts[-1] += 500e6  # last blink is 600 ms (abnormal)
    blinks_df = pd.DataFrame(
        {
            "blink id": np.arange(len(blinks_start_ts)),
            "start timestamp [ns]": blinks_start_ts,
            "end timestamp [ns]": blinks_end_ts,
            "duration [ms]": (blinks_end_ts - blinks_start_ts) / 1e6,
        }
    )

    fixations_start_ts = np.array([0.5e9, 3e9, 7e9])
    fixations_end_ts = fixations_start_ts + 200e6  # 200 ms fixations
    fixations_df = pd.DataFrame(
        {
            "fixation id": np.arange(len(fixations_start_ts)),
            "start timestamp [ns]": fixations_start_ts,
            "end timestamp [ns]": fixations_end_ts,
            "duration [ms]": (fixations_end_ts - fixations_start_ts) / 1e6,
            "fixation x [px]": np.random.rand(len(fixations_start_ts)),
            "fixation y [px]": np.random.rand(len(fixations_start_ts)),
        }
    )

    saccades_start_ts = np.array([2.5e9, 5e9, 8e9])
    saccades_end_ts = saccades_start_ts + 50e6  # 50 ms saccades
    saccades_df = pd.DataFrame(
        {
            "saccade id": np.arange(len(saccades_start_ts)),
            "start timestamp [ns]": saccades_start_ts,
            "end timestamp [ns]": saccades_end_ts,
            "duration [ms]": (saccades_end_ts - saccades_start_ts) / 1e6,
            "amplitude [px]": np.random.rand(len(saccades_start_ts)) * 100,
            "amplitude [deg]": np.random.rand(len(saccades_start_ts)) * 10,
        }
    )

    events_ts = np.array([4e9, 9e9])
    events_names = ["stimulus_onset", "stimulus_offset"]
    events_df = pd.DataFrame(
        {
            "timestamp [ns]": events_ts,
            "name": events_names,
            "type": "recording",
        }
    )

    custom_ts = np.array([1.5e9, 4.5e9])
    custom_names = ["custom_event_1", "custom_event_2"]
    custom_df = pd.DataFrame(
        {
            "timestamp [ns]": custom_ts,
            "custom_name": custom_names,
        }
    )
    print(custom_df)

    blinks = Events(blinks_df)
    assert blinks.type == "blinks"
    assert blinks.data.index.name == "blink id"

    fixations = Events(fixations_df)
    assert fixations.type == "fixations"
    assert fixations.data.index.name == "fixation id"

    saccades = Events(saccades_df)
    assert saccades.type == "saccades"
    assert saccades.data.index.name == "saccade id"

    events = Events(events_df)
    assert events.type == "events"
    assert events.data.index.name == "event id"

    with pytest.warns(UserWarning, match="Following columns not in known data types"):
        custom = Events(custom_df)
    print(custom.data)
    assert custom.type == "custom"
    assert custom.data.index.name == "event id"

    return blinks, fixations, saccades, events, custom
