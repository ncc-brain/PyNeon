from pathlib import Path
from pyneon import NeonRecording
from matplotlib import pyplot as plt

from pyneon import get_sample_data

# all paths are relative to the location of the notebook
dataset_path = get_sample_data('OfficeWalk')
recording_dir = dataset_path / 'Timeseries Data' /'walk1-e116e606'


test_output_dir = 'data/outputs'
test_output_dir.mkdir(exist_ok=True)

recording = NeonRecording(recording_dir)
raw_gaze_data = recording.gaze.data
raw_eye_states_data = recording.eye_states.data
raw_imu_data = recording.imu.data

concat_df = recording.concat_channels(["gaze", "eye_states"])
concat_df.to_csv(test_output_dir / "concat_gaze_eye_states.csv", index=False)
fig, ax = plt.subplots()
ax.plot(
    concat_df["timestamp [ns]"][-100:],
    concat_df["gaze x [px]"][-100:],
    color="blue",
    label="concat",
)
ax.plot(
    raw_gaze_data["timestamp [ns]"][-100:],
    raw_gaze_data["gaze x [px]"][-100:],
    color="red",
    label="raw",
)
ax.legend()
fig.savefig(test_output_dir / "concat_gaze_eye_states.png")

concat_df = recording.concat_channels(["gaze", "eye_states", "IMU"], sampling_freq=1000)
concat_df.to_csv(test_output_dir / "concat_gaze_eye_states_imu.csv", index=False)
fig, ax = plt.subplots()
ax.plot(
    concat_df["timestamp [ns]"][-100:],
    concat_df["gaze x [px]"][-100:],
    color="blue",
    label="concat",
)
ax.plot(
    raw_gaze_data["timestamp [ns]"][-200:],
    raw_gaze_data["gaze x [px]"][-200:],
    color="red",
    label="raw",
)
ax.legend()
fig.savefig(test_output_dir / "concat_gaze_eye_states_imu.png")
