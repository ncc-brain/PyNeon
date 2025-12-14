from pyneon.utils.variables import nominal_sampling_rates


def test_metadata(simple_streams):
    gaze, eye_states, imu, custom = simple_streams

    assert gaze.type == "gaze"
    assert gaze.sampling_freq_nominal == nominal_sampling_rates["gaze"]

    assert eye_states.type == "eye_states"
    assert eye_states.sampling_freq_nominal == nominal_sampling_rates["eye_states"]

    assert imu.type == "imu"
    assert imu.sampling_freq_nominal == nominal_sampling_rates["imu"]

    assert custom.type == "custom"
    assert custom.sampling_freq_nominal is None
