import numpy as np
import pytest


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
def test_interp(simple_streams, float_kind):
    gaze, _, _, custom = simple_streams

    # Before interpolation, the time differences should not match the nominal sampling period
    assert not np.isclose(gaze.ts_diff.mean(), 1e9 / gaze.sampling_freq_nominal)

    # Default interpolation
    new_gaze = gaze.interpolate(float_kind=float_kind)

    # After interpolation, the time differences should match the nominal sampling period
    assert np.isclose(new_gaze.ts_diff.mean(), 1e9 / new_gaze.sampling_freq_nominal)

    # When inplace=True, the original stream should be modified and return None
    gaze_copy = gaze.copy()
    result = gaze_copy.interpolate(float_kind=float_kind, inplace=True)
    assert result is None
    assert np.isclose(gaze_copy.ts_diff.mean(), 1e9 / gaze_copy.sampling_freq_nominal)

    # Interpolate custom stream without new_ts should raise an ValueError
    with pytest.raises(ValueError):
        custom.interpolate(float_kind=float_kind)


def test_simple_concat(simple_streams):
    gaze, eye_states, imu, custom = simple_streams

    # custom has a larger temporal range, so gaze.concat(custom) should not trigger any warning
    concat = gaze.concat(custom)
    assert np.array_equal(concat.ts, gaze.ts)
    assert list(concat.columns) == gaze.columns.tolist() + custom.columns.tolist()

    # eye_states starts later than gaze, so gaze.concat(eye_states) should trigger a warning
    with pytest.warns(
        UserWarning, match="new_ts contains timestamps before the data start time"
    ):
        concat = gaze.concat(eye_states)
    assert np.array_equal(concat.ts, gaze.ts)
    assert list(concat.columns) == gaze.columns.tolist() + eye_states.columns.tolist()

    # imu ends earlier than gaze, so gaze.concat(imu) should trigger a warning
    with pytest.warns(
        UserWarning, match="new_ts contains timestamps after the data end time"
    ):
        concat = gaze.concat(imu)
    assert np.array_equal(concat.ts, gaze.ts)
    assert list(concat.columns) == gaze.columns.tolist() + imu.columns.tolist()