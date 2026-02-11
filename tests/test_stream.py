import re

import numpy as np
import pytest


@pytest.mark.parametrize(
    "by",
    ["timestamp", "time", "sample"],
)
def test_crop(sim_gaze, by):
    ts0 = sim_gaze.ts
    if by == "timestamp":
        t0 = ts0.copy()
    elif by == "time":
        t0 = sim_gaze.times
    else:
        t0 = np.arange(len(sim_gaze))
    tmax_index = len(t0) // 2
    ts_first_half = ts0[:tmax_index]

    sim_gaze_cropped = sim_gaze.crop(tmax=t0[tmax_index], by=by)
    assert np.array_equal(sim_gaze_cropped.ts, ts_first_half)

    # If none of tmin and tmax is provided, should raise ValueError
    with pytest.raises(
        ValueError, match=re.escape("At least one of `tmin` or `tmax` must be provided")
    ):
        sim_gaze.crop(by=by)

    # If cropping after the end time, should find no data and raise ValueError
    with pytest.raises(ValueError, match="No data found in the specified time range"):
        sim_gaze.crop(tmin=t0[-1] + 1e9, by=by)


@pytest.mark.parametrize(
    "float_kind",
    [
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ],
)
def test_interp(sim_gaze, sim_custom_stream, float_kind):
    # Before interpolation, the time differences should not match the nominal sampling period
    # and the stream should not be uniformly sampled
    assert not np.isclose(sim_gaze.ts_diff.mean(), 1e9 / sim_gaze.sampling_freq_nominal)
    assert not sim_gaze.is_uniformly_sampled

    # Default interpolation
    new_gaze = sim_gaze.interpolate(float_kind=float_kind)

    # After interpolation, the time differences should match the nominal sampling period
    assert np.isclose(new_gaze.ts_diff.mean(), 1e9 / new_gaze.sampling_freq_nominal)
    assert new_gaze.is_uniformly_sampled

    # When inplace=True, the original stream should be modified and return None
    gaze_copy = sim_gaze.copy()
    result = gaze_copy.interpolate(float_kind=float_kind, inplace=True)
    assert result is None
    assert np.isclose(gaze_copy.ts_diff.mean(), 1e9 / gaze_copy.sampling_freq_nominal)
    assert gaze_copy.is_uniformly_sampled

    # Interpolate custom stream without new_ts should raise an ValueError
    with pytest.raises(ValueError):
        sim_custom_stream.interpolate(float_kind=float_kind)


def test_simple_concat(sim_gaze, sim_eye_states, sim_imu, sim_custom_stream):
    # custom has a larger temporal range, so gaze.concat(custom) should not trigger any warning
    concat = sim_gaze.concat(sim_custom_stream)
    assert np.array_equal(concat.ts, sim_gaze.ts)
    assert (
        list(concat.columns)
        == sim_gaze.columns.tolist() + sim_custom_stream.columns.tolist()
    )

    # eye_states starts later than gaze, so gaze.concat(eye_states) should trigger a warning
    with pytest.warns(
        UserWarning, match="requested timestamps are outside the data time range"
    ):
        concat = sim_gaze.concat(sim_eye_states)
    assert np.array_equal(concat.ts, sim_gaze.ts)
    assert (
        list(concat.columns)
        == sim_gaze.columns.tolist() + sim_eye_states.columns.tolist()
    )

    # imu ends earlier than gaze, so gaze.concat(imu) should trigger a warning
    with pytest.warns(
        UserWarning, match="requested timestamps are outside the data time range"
    ):
        concat = sim_gaze.concat(sim_imu)
    assert np.array_equal(concat.ts, sim_gaze.ts)
    assert list(concat.columns) == sim_gaze.columns.tolist() + sim_imu.columns.tolist()
