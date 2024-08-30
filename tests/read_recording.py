from pathlib import Path
from pyneon import NeonRecording

recording_dir = (
    Path(__file__).parent / "data" / "Timeseries Data + Scene Video" / "walk2-93b8c234"
)
recording = NeonRecording(recording_dir)

# Should print the contents of the recording
print(recording)

# Should print the gaze dataframe
print(recording.gaze().data)
