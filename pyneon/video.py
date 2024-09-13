import cv2
import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt


class NeonVideo(cv2.VideoCapture):
    """
    Loaded video file with timestamps.

    Parameters
    ----------
    video_file : :class:`pathlib.Path`
        Path to the video file.
    timestamps_file : :class:`pathlib.Path`
        Path to the timestamps file.

    Attributes
    ----------
    timestamps : np.ndarray
        Timestamps of the video frames.
    ts : np.ndarray
        Alias for timestamps.
    n_frames : int
        Number of frames in the video.
    fps : float
        Frames per second of the video.
    width : int
        Width of the video frames in pixels.
    height : int
        Height of the video frames in pixels.
    """

    def __init__(self, video_file, timestamps_file):
        super().__init__(video_file)
        self.timestamps = (
            pd.read_csv(timestamps_file)["timestamp [ns]"].to_numpy().astype(np.int64)
        )
        self.ts = self.timestamps
        assert len(self.timestamps) == self.get(cv2.CAP_PROP_FRAME_COUNT), (
            f"Number of timestamps ({len(self.timestamps)}) does not match "
            f"number of frames ({self.get(cv2.CAP_PROP_FRAME_COUNT)})"
        )

        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self) -> int:
        return int(len(self.ts))


def plot_frame(self, index: int = 0, ax: Union[plt.Axes, None] = None):
    """
    Plot a frame from the video.

    Parameters
    ----------
    index : int
        Index of the frame to plot.
    """
    if index >= len(self.ts):
        raise IndexError(f"Frame index {index} out of range")

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    self.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = self.read()
    if ret:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {index} ({self.ts[index]})")
        ax.axis("off")
    else:
        raise RuntimeError(f"Could not read frame {index}")
