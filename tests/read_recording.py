from pathlib import Path
from pyneon import NeonRecording
import numpy as np


recording_dir = (
    Path(__file__).parent / "data" / "Timeseries Data + Scene Video" / "walk2-93b8c234"
)
recording = NeonRecording(recording_dir)
gaze = recording.gaze

# Should print the contents of the recording
print(recording)
dtype = gaze.data.dtypes["worn"].type
print(dtype)
print(np.issubdtype(dtype, np.integer))
# # # Should print the gaze dataframe
# # # print(recording.imu.data)

resamp_data = gaze.resample()
resamp_data.to_csv("resampled_gaze.csv", index=False)

print(gaze.first_ts)
new_ts = np.arange(
    gaze.first_ts, gaze.last_ts, 1e9 / gaze.sampling_rate_nominal, dtype=np.int64
)
print(new_ts[0])
