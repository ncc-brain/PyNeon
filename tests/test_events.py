import re

import numpy as np
import pytest


@pytest.mark.parametrize(
    "by",
    ["timestamp", "sample"],
)
def test_crop(sim_blinks, by):
    ts0 = sim_blinks.start_ts
    if by == "timestamp":
        t0 = ts0.copy()
    else:
        t0 = np.arange(len(sim_blinks))
    tmax_index = len(t0) // 2
    ts_first_half = ts0[: tmax_index + 1]

    sim_blinks_cropped = sim_blinks.crop(tmax=t0[tmax_index], by=by)
    assert np.array_equal(sim_blinks_cropped.start_ts, ts_first_half)

    # If none of tmin and tmax is provided, should raise ValueError
    with pytest.raises(
        ValueError, match=re.escape("At least one of `tmin` or `tmax` must be provided")
    ):
        sim_blinks.crop(by=by)

    # If cropping after the end time, should find no data and raise ValueError
    with pytest.raises(ValueError, match="No data found in the specified time range"):
        sim_blinks.crop(tmin=t0[-1] + 1e9, by=by)


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
