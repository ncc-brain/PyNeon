import numpy as np
import pytest


@pytest.mark.parametrize(
    "kind",
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
def test_interp(simple_streams, kind):
    gaze, _, _, custom = simple_streams

    # Before interpolation, the time differences should not match the nominal sampling period
    assert not np.isclose(gaze.ts_diff.mean(), 1e9 / gaze.sampling_freq_nominal)

    # Default interpolation
    new_gaze = gaze.interpolate(float_kind=kind)

    # After interpolation, the time differences should match the nominal sampling period
    assert np.isclose(new_gaze.ts_diff.mean(), 1e9 / new_gaze.sampling_freq_nominal)

    # When inplace=True, the original stream should be modified and return None
    gaze_copy = gaze.copy()
    result = gaze_copy.interpolate(float_kind=kind, inplace=True)
    assert result is None
    assert np.isclose(gaze_copy.ts_diff.mean(), 1e9 / gaze_copy.sampling_freq_nominal)

    # Interpolate custom stream without new_ts should raise an ValueError
    with pytest.raises(ValueError):
        custom.interpolate(float_kind=kind)
