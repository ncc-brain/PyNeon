import re

import numpy as np
import pytest

from pyneon import Stream

@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
def test_save_stream(request, dataset_fixture, tmp_path):
    dataset = request.getfixturevalue(dataset_fixture)
    for rec in dataset.recordings:
        for name in ["gaze", "eye_states", "imu"]:
            stream = getattr(rec, name)
            output_path = tmp_path / f"{name}.csv"
            stream.save(output_path)
            assert output_path.exists()
            stream_loaded = Stream(output_path)
            assert np.array_equal(stream_loaded.data.index, stream.data.index)
            for col in stream.columns:
                assert np.allclose(stream_loaded[col], stream[col], equal_nan=True)
            # Delete the saved file after test
            output_path.unlink()

@pytest.mark.parametrize(
    "gaze_fixture",
    ["sim_gaze", "cloud_gaze", "native_gaze"],
)
@pytest.mark.parametrize(
    "by",
    ["timestamp", "time", "sample"],
)
def test_crop(request, gaze_fixture, by):
    gaze = request.getfixturevalue(gaze_fixture)
    ts0 = gaze.ts
    if by == "timestamp":
        t0 = ts0.copy()
    elif by == "time":
        t0 = gaze.times
    else:
        t0 = np.arange(len(gaze))
    tmax_index = len(t0) // 2
    # Cropping is inclusive of tmax, so we need to include the tmax_index in the expected result
    ts_first_half = ts0[: tmax_index + 1]

    gaze_cropped = gaze.crop(tmax=t0[tmax_index], by=by)
    assert np.array_equal(gaze_cropped.ts, ts_first_half)

    # If none of tmin and tmax is provided, should raise ValueError
    with pytest.raises(
        ValueError, match=re.escape("At least one of `tmin` or `tmax` must be provided")
    ):
        gaze.crop(by=by)

    # If cropping after the end time, should find no data and raise ValueError
    with pytest.raises(ValueError, match="No data found in the specified time range"):
        gaze.crop(tmin=t0[-1] + 1e9, by=by)


@pytest.mark.parametrize(
    "gaze_fixture",
    ["sim_gaze", "cloud_gaze", "native_gaze"],
)
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
def test_interp(request, gaze_fixture, sim_custom_stream, float_kind):
    gaze = request.getfixturevalue(gaze_fixture)
    # Before interpolation, the time differences should not match the nominal sampling period
    # and the stream should not be uniformly sampled
    assert not np.isclose(gaze.ts_diff.mean(), 1e9 / gaze.sampling_freq_nominal)
    assert not gaze.is_uniformly_sampled

    # Default interpolation
    new_gaze = gaze.interpolate(float_kind=float_kind)

    # After interpolation, the time differences should match the nominal sampling period
    assert np.isclose(new_gaze.sampling_freq_effective, new_gaze.sampling_freq_nominal)
    assert new_gaze.is_uniformly_sampled

    # When inplace=True, the original stream should be modified and return None
    gaze_copy = gaze.copy()
    result = gaze_copy.interpolate(float_kind=float_kind, inplace=True)
    assert result is None
    assert np.isclose(
        gaze_copy.sampling_freq_effective, gaze_copy.sampling_freq_nominal
    )
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


@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
@pytest.mark.parametrize(
    "stream_names",
    (
        "all",
        ["gaze", "eye_states"],
        ["gaze", "eye_states", "imu"],
        "xxx",
        ["xxx", "gaze"],
    ),
)
@pytest.mark.parametrize(
    "sampling_freq",
    ("min", "max", 150),
)
def test_concat(request, dataset_fixture, stream_names, sampling_freq):
    dataset = request.getfixturevalue(dataset_fixture)
    for rec in dataset.recordings:
        # First test Recording-level concatenation
        if stream_names == "xxx":
            with pytest.raises(
                ValueError,
                match="Invalid stream_names, must be 'all' or a list of stream names.",
            ):
                concat_stream = rec.concat_streams(
                    stream_names=stream_names, sampling_freq=sampling_freq
                )
            break
        elif "xxx" in stream_names:
            with pytest.raises(
                ValueError, match="Invalid stream name, can only be one of"
            ):
                concat_stream = rec.concat_streams(
                    stream_names=stream_names, sampling_freq=sampling_freq
                )
            break
        else:
            concat_stream = rec.concat_streams(
                stream_names=stream_names, sampling_freq=sampling_freq
            )
        stream_names = (
            ["gaze", "eye_states", "imu"] if stream_names == "all" else stream_names
        )
        expected_columns = []
        samp_freq_list = []
        for name in stream_names:
            expected_columns += getattr(rec, name).columns.tolist()
            samp_freq_list.append(getattr(rec, name).sampling_freq_nominal)
        assert set(concat_stream.columns) == set(expected_columns)
        if sampling_freq == "min":
            assert np.isclose(
                concat_stream.sampling_freq_effective, min(samp_freq_list)
            )
        elif sampling_freq == "max":
            assert np.isclose(
                concat_stream.sampling_freq_effective, max(samp_freq_list)
            )
        else:
            assert np.isclose(concat_stream.sampling_freq_effective, sampling_freq)
