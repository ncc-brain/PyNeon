import re

import numpy as np
import pytest

from pyneon import Events


@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
def test_save_events(request, dataset_fixture, tmp_path):
    dataset = request.getfixturevalue(dataset_fixture)
    for rec in dataset.recordings:
        for name in ["blinks", "fixations", "saccades", "events"]:
            try:
                events = getattr(rec, name)
            except ValueError:  # Empty data
                continue
            output_path = tmp_path / f"{name}.csv"
            events.save(output_path)
            assert output_path.exists()
            events_loaded = Events(output_path)
            assert np.array_equal(events_loaded.data.index, events.data.index)
            for col in events.columns:
                # assert same for col of str type
                if events.data[col].dtype == "string":
                    assert np.array_equal(events_loaded.data[col], events.data[col])
                else:
                    # For numeric columns, use np.allclose to handle potential floating point differences
                    assert np.allclose(
                        events_loaded.data[col], events.data[col], equal_nan=True
                    ), f"Column '{col}' does not match after loading from CSV."
            # Delete the saved file after test
            output_path.unlink()


@pytest.mark.parametrize(
    "fixations_fixture",
    ["sim_fixations", "cloud_fixations", "native_fixations"],
)
@pytest.mark.parametrize(
    "by",
    ["timestamp", "sample"],
)
def test_crop(request, fixations_fixture, by):
    fixations = request.getfixturevalue(fixations_fixture)
    ts0 = fixations.start_ts
    if by == "timestamp":
        t0 = ts0.copy()
    else:
        t0 = np.arange(len(fixations))
    tmax_index = len(t0) // 2
    # Cropping is inclusive of tmax, so we need to include the tmax_index in the expected result
    ts_first_half = ts0[: tmax_index + 1]

    fixations_cropped = fixations.crop(tmax=t0[tmax_index], by=by)
    assert np.array_equal(fixations_cropped.start_ts, ts_first_half)

    # If none of tmin and tmax is provided, should raise ValueError
    with pytest.raises(
        ValueError, match=re.escape("At least one of `tmin` or `tmax` must be provided")
    ):
        fixations.crop(by=by)

    # If cropping after the end time, should find no data and raise ValueError
    with pytest.raises(ValueError, match="No data found in the specified time range"):
        fixations.crop(tmin=t0[-1] + 1e9, by=by)


def test_filter_by_duration(sim_blinks):
    # Check that blinks have expected durations before filtering (50 ms, 100 ms, 500 ms)
    assert set(sim_blinks.durations) == {50, 100, 500}
    n_blinks = len(sim_blinks)

    # Filter blinks with duration >= 100 ms
    filtered_blinks = sim_blinks.filter_by_duration(dur_min=100)
    assert 50 not in filtered_blinks.durations
    assert 500 in filtered_blinks.durations
    assert len(filtered_blinks) == n_blinks - 1

    # Test inplace
    blinks_copy = sim_blinks.copy()
    result = blinks_copy.filter_by_duration(dur_min=100, inplace=True)
    assert result is None
    assert 50 not in blinks_copy.durations
    assert len(blinks_copy) == n_blinks - 1

    # Filter blinks with duration <= 400 ms
    filtered_blinks = sim_blinks.filter_by_duration(dur_max=400)
    assert 50 in filtered_blinks.durations
    assert 500 not in filtered_blinks.durations
    assert len(filtered_blinks) == n_blinks - 1

    # Filter blinks with duration between 100 ms and 400 ms
    filtered_blinks = sim_blinks.filter_by_duration(dur_min=100, dur_max=400)
    assert 50 not in filtered_blinks.durations
    assert 500 not in filtered_blinks.durations
    assert len(filtered_blinks) == n_blinks - 2

    # Filter blinks with duration between 100 ms and 500 ms (both inclusive)
    filtered_blinks = sim_blinks.filter_by_duration(dur_min=100, dur_max=500)
    assert 50 not in filtered_blinks.durations
    assert 500 in filtered_blinks.durations
    assert len(filtered_blinks) == n_blinks - 1


def test_filter_by_name(sim_events):
    # Assert initial event names
    expected_names = [
        "recording.begin",
        "stimulus_onset",
        "stimulus_offset",
        "recording.end",
    ]
    assert all(name in expected_names for name in sim_events.data["name"])

    # Filter events with name 'stimulus_onset'
    filtered_events = sim_events.filter_by_name(names=["stimulus_onset"])
    assert all(name == "stimulus_onset" for name in filtered_events.data["name"])
    assert len(filtered_events) == 1

    # Filter events with names 'stimulus_onset' and 'stimulus_offset'
    filtered_events = sim_events.filter_by_name(
        names=["stimulus_onset", "stimulus_offset"]
    )
    assert all(
        name in ["stimulus_onset", "stimulus_offset"]
        for name in filtered_events.data["name"]
    )
    assert len(filtered_events) == 2

    # Test inplace
    events_copy = sim_events.copy()
    result = events_copy.filter_by_name(names=["stimulus_onset"], inplace=True)
    assert result is None
    assert all(name == "stimulus_onset" for name in events_copy.data["name"])
    assert len(events_copy) == 1


@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
@pytest.mark.parametrize(
    "events_names",
    (
        "all",
        ["fixations", "saccades"],
        ["fixations", "saccades", "blinks"],
        ["fixations", "saccades", "events"],
        "xxx",
        ["xxx", "fixations"],
    ),
)
def test_concat(request, dataset_fixture, events_names):
    dataset = request.getfixturevalue(dataset_fixture)
    for rec in dataset.recordings:
        # First test Recording-level concatenation
        if events_names == "xxx":
            with pytest.raises(
                ValueError,
                match="Invalid events_names, must be 'all' or a list of event names.",
            ):
                concat_events = rec.concat_events(events_names=events_names)
            break
        if "xxx" in events_names:
            with pytest.raises(
                ValueError, match="Invalid event name, can only be one of"
            ):
                concat_events = rec.concat_events(events_names=events_names)
            break
        if len(events_names) == 1:
            with pytest.raises(
                ValueError,
                match="Must provide at least two events to concatenate",
            ):
                concat_events = rec.concat_events(events_names=events_names)
            break
        if "blinks" in events_names:
            try:  # Test if blinks can be obtained
                _ = rec.blinks
            except Exception:
                break
        concat_events = rec.concat_events(events_names=events_names)
        events_names = (
            ["blinks", "fixations", "saccades", "events"]
            if events_names == "all"
            else events_names
        )
        expected_n_events = 0
        expected_columns = [
            "type"
        ]  # Always have a "type" column to indicate the event type
        for name in events_names:
            expected_n_events += len(getattr(rec, name))
            new_columns = getattr(rec, name).columns.tolist()
            if name == "events":
                new_columns.remove("timestamp [ns]")
                new_columns = [
                    col if col != "name" else "message name" for col in new_columns
                ]
                new_columns = [
                    col if col != "type" else "message type" for col in new_columns
                ]
            expected_columns += new_columns
        assert set(concat_events.columns) == set(expected_columns)
        assert len(concat_events) == expected_n_events
