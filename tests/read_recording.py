from pathlib import Path
from pyneon import NeonRecording
import numpy as np


recording_dir = (
    Path(__file__).parent / "data" / "Timeseries Data + Scene Video" / "walk2-93b8c234"
)
recording = NeonRecording(recording_dir)
print(recording)
