import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import json

from ..vis import plot_frame, plot_scanpath_on_video


class NeonVideo(cv2.VideoCapture):
    """
    Loaded video file with timestamps.

    Parameters
    ----------
    video_file : :class:`pathlib.Path`
        Path to the video file.
    timestamps_file : :class:`pathlib.Path`
        Path to the timestamps file.
    info_file : :class:`pathlib.Path`
        Path to the scene camera info file.

    Attributes
    ----------
    timestamps : np.ndarray
        Timestamps of the video frames in nanoseconds.
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

    def __init__(self, video_file: Path, timestamps_file: Path, info_file: Path):
        super().__init__(video_file)
        self.video_file = video_file
        self.timestamps_file = timestamps_file
        self.info_file = info_file
        with open(info_file) as f:
            self.info = json.load(f)
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

        self.camera_matrix = np.array(self.info["camera_matrix"])
        self.dist_coeffs = np.array(self.info["distortion_coefficients"])

    def __len__(self) -> int:
        return int(len(self.ts))

    def plot_frame(
        self,
        index: int = 0,
        ax: Optional[plt.Axes] = None,
        auto_title: bool = True,
        show: bool = True,
    ):
        """
        Plot a frame from the video on a matplotlib axis.

        Parameters
        ----------
        index : int
            Index of the frame to plot.
        ax : :class:`matplotlib.pyplot.Axes` or None
            Axis to plot the frame on. If ``None``, a new figure is created.
            Defaults to ``None``.
        auto_title : bool
            Whether to automatically set the title of the axis.
            The automatic title includes the video file name and the frame index.
            Defaults to ``True``.

        Returns
        -------
        fig : :class:`matplotlib.pyplot.Figure`
            Figure object containing the plot.
        ax : :class:`matplotlib.pyplot.Axes`
            Axis object containing the plot.
        """
        return plot_frame(self, index, ax, auto_title, show)

    def plot_scanpath_on_video(
        self,
        scanpath: pd.DataFrame,
        circle_radius: int = 10,
        line_thickness: int = 2,
        max_fixations: int = 10,
        show_video: bool = False,
        video_output_path: Path | str = "scanpath.mp4",
    ) -> None:
        """
        Plot scanpath on top of the video frames. The resulting video can be displayed and/or saved.

        Parameters
        ----------
        scanpath : :class:`pandas.DataFrame`
            DataFrame containing the fixations and gaze data.
        circle_radius : int
            Radius of the fixation circles in pixels. Defaults to 10.
        line_thickness : int or None
            Thickness of the lines connecting fixations. If None, no lines are drawn.
            Defaults to 2.
        max_fixations : int
            Maximum number of fixations to plot per frame. Defaults to 10.
        show_video : bool
            Whether to display the video with fixations overlaid. Defaults to False.
        video_output_path : :class:`pathlib.Path` or str or None
            Path to save the video with fixations overlaid. If None, the video is not saved.
            Defaults to 'scanpath.mp4'.
        """
        plot_scanpath_on_video(
            self,
            scanpath,
            circle_radius,
            line_thickness,
            max_fixations,
            show_video,
            video_output_path,
        )
