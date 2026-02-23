import re

import numpy as np
import pandas as pd
import pytest

from pyneon import Dataset, Events, Stream, get_sample_data
from pyneon.utils.variables import nominal_sampling_rates


@pytest.fixture(scope="package")
def sim_gaze():
    ts = np.arange(1e9, 5e9, 1e9 / 50)  # 50 Hz
    ts = np.delete(ts, 2)  # Remove one ts to make it non-uniformly sampled
    df = pd.DataFrame(
        np.random.rand(len(ts), 2),
        index=ts,
        columns=["gaze x [px]", "gaze y [px]"],
    )
    df.index.name = "timestamp [ns]"

    gaze = Stream(df)
    assert gaze.type == "gaze"
    assert gaze.sampling_freq_nominal == nominal_sampling_rates["gaze"]
    return gaze


@pytest.fixture(scope="package")
def sim_imu():
    ts = np.arange(1e9, 4e9, 1e9 / 200)  # 200 Hz
    ts = np.delete(ts, 2)  # Remove one ts to make it non-uniformly sampled
    df = pd.DataFrame(
        np.random.rand(len(ts), 2),
        index=ts,
        columns=["gyro x [deg/s]", "gyro y [deg/s]"],
    )
    df.index.name = "timestamp [ns]"

    imu = Stream(df)
    assert imu.type == "imu"
    assert imu.sampling_freq_nominal == nominal_sampling_rates["imu"]
    return imu


@pytest.fixture(scope="package")
def sim_eye_states():
    ts = np.arange(2e9, 6e9, 1e9 / 100)  # 100 Hz
    ts = np.delete(ts, 2)  # Remove one ts to make it non-uniformly sampled
    df = pd.DataFrame(
        np.random.rand(len(ts), 2),
        index=ts,
        columns=["pupil diameter left [mm]", "pupil diameter right [mm]"],
    )
    df.index.name = "timestamp [ns]"

    eye_states = Stream(df)
    assert eye_states.type == "eye_states"
    assert eye_states.sampling_freq_nominal == nominal_sampling_rates["eye_states"]
    return eye_states


@pytest.fixture(scope="package")
def sim_custom_stream():
    ts = np.arange(0, 6e9, int(1e9 / 30))  # 30 Hz
    df = pd.DataFrame(
        np.random.rand(len(ts), 2),
        index=ts,
        columns=["custom x", "custom y"],
    )
    df.index.name = "timestamp [ns]"

    with pytest.warns(UserWarning, match="Following columns not in known data types"):
        custom = Stream(df)
    assert custom.type == "custom"
    assert custom.sampling_freq_nominal is None
    return custom


@pytest.fixture(scope="package")
def sim_blinks():
    blinks_start_ts = np.array([1e9, 3e9, 5e9, 8e9, 10e9])
    blinks_end_ts = blinks_start_ts + 100e6  # 100 ms blinks
    blinks_end_ts[0] = blinks_start_ts[0] + 50e6  # first blink is 50 ms (abnormal)
    blinks_end_ts[-1] = blinks_start_ts[-1] + 500e6  # last blink is 500 ms (abnormal)
    blinks_df = pd.DataFrame(
        {
            "blink id": np.arange(len(blinks_start_ts)),
            "start timestamp [ns]": blinks_start_ts,
            "end timestamp [ns]": blinks_end_ts,
            "duration [ms]": (blinks_end_ts - blinks_start_ts) / 1e6,
        }
    )

    blinks = Events(blinks_df)
    assert blinks.type == "blinks"
    assert blinks.data.index.name == "blink id"
    return blinks


@pytest.fixture(scope="package")
def sim_fixations():
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

    fixations = Events(fixations_df)
    assert fixations.type == "fixations"
    assert fixations.data.index.name == "fixation id"
    return fixations


@pytest.fixture(scope="package")
def sim_saccades():
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

    saccades = Events(saccades_df)
    assert saccades.type == "saccades"
    assert saccades.data.index.name == "saccade id"
    return saccades


@pytest.fixture(scope="package")
def sim_events():
    events_ts = np.array([0, 4e9, 9e9, 10e9])
    events_names = [
        "recording.begin",
        "stimulus_onset",
        "stimulus_offset",
        "recording.end",
    ]
    events_df = pd.DataFrame(
        {
            "timestamp [ns]": events_ts,
            "name": events_names,
            "type": "recording",
        }
    )

    events = Events(events_df)
    assert events.type == "events"
    assert events.data.index.name == "event id"
    with pytest.raises(
        ValueError, match=re.escape("No `duration [ms]` column found in the instance.")
    ):
        _ = events.durations
    with pytest.raises(
        ValueError,
        match=re.escape("No `end timestamp [ns]` column found in the instance."),
    ):
        _ = events.end_ts
    return events


@pytest.fixture(scope="package")
def sim_custom_events():
    custom_ts = np.array([1.5e9, 4.5e9])
    custom_names = ["custom_event_1", "custom_event_2"]
    custom_df = pd.DataFrame(
        {
            "timestamp [ns]": custom_ts,
            "custom_name": custom_names,
        }
    )

    with pytest.warns(UserWarning, match="Following columns not in known data types"):
        custom = Events(custom_df)
    assert custom.type == "custom"
    assert custom.data.index.name == "event id"
    return custom


@pytest.fixture(scope="package")
def simple_dataset_native():
    dataset_dir = get_sample_data("simple", format="native")
    dataset = Dataset(dataset_dir)
    assert len(dataset) == dataset.sections.shape[0] == 2
    for recording in dataset.recordings:
        try:
            _ = recording.blinks
            recording.export_cloud_format("data/export", rebase=False)
        except ValueError:
            with pytest.warns(UserWarning, match=r"'blinks' data is empty"):
                recording.export_cloud_format("data/export", rebase=False)
    yield dataset
    for recording in dataset.recordings:
        recording.close()


@pytest.fixture(scope="package")
def simple_dataset_cloud():
    dataset_dir = get_sample_data("simple", format="cloud")
    dataset = Dataset(dataset_dir)
    assert len(dataset) == dataset.sections.shape[0] == 2
    for recording in dataset.recordings:
        with pytest.raises(ValueError, match="Recording is already in Cloud format"):
            recording.export_cloud_format("data/export", rebase=False)
    yield dataset
    for recording in dataset.recordings:
        recording.close()


@pytest.fixture(scope="package")
def cloud_gaze(simple_dataset_cloud):
    return simple_dataset_cloud.recordings[0].gaze


@pytest.fixture(scope="package")
def native_gaze(simple_dataset_native):
    return simple_dataset_native.recordings[0].gaze


@pytest.fixture(scope="package")
def cloud_fixations(simple_dataset_cloud):
    return simple_dataset_cloud.recordings[0].fixations


@pytest.fixture(scope="package")
def native_fixations(simple_dataset_native):
    return simple_dataset_native.recordings[0].fixations
