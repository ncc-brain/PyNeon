from pathlib import Path
from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from ..utils.variables import default_camera_info
from ..vis import overlay_scanpath, plot_frame
from .marker_mapping import detect_markers


class Video(cv2.VideoCapture):
    """
    Loaded video file with timestamps.

    Parameters
    ----------
    video_file : pathlib.Path
        Path to the video file.
    timestamps : numpy.ndarray
        Timestamps of the video frames in nanoseconds.
        Must have the same length as the number of frames in the video.
    info : dict
        Dictionary containing video info, including camera matrix and distortion coefficients.

    Attributes
    ----------
    timestamps : numpy.ndarray
        Timestamps of the video frames in nanoseconds.
    ts : numpy.ndarray
        Alias for timestamps.
    """

    def __init__(self, video_file: Path, timestamps: np.ndarray, info: dict):
        super().__init__(video_file)
        self.video_file = video_file
        self.timestamps = timestamps
        self.ts = self.timestamps
        self.info = info

        if not info:
            warn("Video info is empty and will be loaded from default values.")
            self.info = default_camera_info

        if len(self.timestamps) != self.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError(
                f"Number of timestamps ({len(self.timestamps)}) does not match "
                f"number of frames ({self.get(cv2.CAP_PROP_FRAME_COUNT)})"
            )

    def __len__(self) -> int:
        return int(len(self.ts))

    @property
    def first_ts(self) -> int:
        """First timestamp of the stream."""
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Last timestamp of the stream."""
        return int(self.ts[-1])

    @property
    def ts_diff(self) -> np.ndarray:
        """Difference between consecutive timestamps."""
        return np.diff(self.ts)

    @property
    def times(self) -> np.ndarray:
        """Timestamps converted to seconds relative to video start."""
        return (self.ts - self.ts[0]) / 1e9

    @property
    def duration(self) -> float:
        """Duration of the stream in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Width of the video frames in pixels."""
        return int(self.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Height of the video frames in pixels."""
        return int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def camera_matrix(self) -> np.ndarray:
        """Camera matrix of the video camera."""
        if "camera_matrix" not in self.info:
            raise ValueError("Camera matrix not found in video info.")
        return np.array(self.info["camera_matrix"])

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Distortion coefficients of the video camera."""
        if "distortion_coefficients" not in self.info:
            raise ValueError("Distortion coefficients not found in video info.")
        return np.array(self.info["distortion_coefficients"])

    def get_frame(self, timestamp: Union[int, np.int64]) -> int:
        """
        Get the frame index corresponding to a given timestamp.

        Parameters
        ----------
        timestamp : int
            Timestamp in nanoseconds.

        Returns
        -------
        int
            Frame index corresponding to the timestamp.
        """
        if timestamp < self.ts[0] or timestamp > self.ts[-1]:
            raise ValueError("Timestamp is out of bounds of the video timestamps.")

        return int(np.searchsorted(self.ts, timestamp))

    def reset(self):
        if self.isOpened():
            self.release()
        super().__init__(self.video_file)
        self.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not self.isOpened():
            raise IOError(f"Failed to reopen video file: {self.video_file}")

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
        ax : matplotlib.axes.Axes or None
            Axis to plot the frame on. If ``None``, a new figure is created.
            Defaults to ``None``.
        auto_title : bool
            Whether to automatically set the title of the axis.
            The automatic title includes the video file name and the frame index.
            Defaults to ``True``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.
        ax : matplotlib.axes.Axes
            Axis object containing the plot.
        """
        return plot_frame(self, index, ax, auto_title, show)

    @fill_doc
    def detect_markers(
        self,
        marker_name: str = "36h11",
        step: int = 1,
        detection_window: Optional[tuple[int | float, int | float]] = None,
        detection_window_unit: str = "frame",
    ) -> Stream:
        """
        Detect fiducial markers (AprilTag or ArUco) in the video frames.

        Parameters
        ----------
        %(detect_markers_params)s

        Returns
        -------
        %(detect_markers_return)s
        """
        return detect_markers(
            self,
            marker_name=marker_name,
            step=step,
            detection_window=detection_window,
            detection_window_unit=detection_window_unit,
        )

    def overlay_scanpath(
        self,
        scanpath: pd.DataFrame,
        circle_radius: int = 10,
        line_thickness: int = 2,
        max_fixations: int = 10,
        show_video: bool = False,
        video_output_path: Path | str = "derivatives/scanpath.mp4",
    ) -> None:
        """
        Plot scanpath on top of the video frames. The resulting video can be displayed and/or saved.

        Parameters
        ----------
        scanpath : pandas.DataFrame
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
        video_output_path : pathlib.Path or str or None
            Path to save the video with fixations overlaid. If None, the video is not saved.
            Defaults to 'scanpath.mp4'.
        """
        overlay_scanpath(
            self,
            scanpath,
            circle_radius,
            line_thickness,
            max_fixations,
            show_video,
            video_output_path,
        )

    def undistort(
        self,
        output_video_path: Optional[Path | str] = "undistorted_video.mp4",
    ) -> None:
        """
        Undistort a video using the known camera matrix and distortion coefficients.

        Parameters
        ----------
        output_video_path : str
            Path to save the undistorted output video.
        """
        # Open the input video
        cap = self
        cap.reset()
        camera_matrix = self.camera_matrix
        dist_coeffs = self.distortion_coefficients

        # Get self properties
        frame_width, frame_height = self.width, self.height
        fps = self.fps
        frame_count = len(self)

        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Adjust codec as needed
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

        # Precompute the optimal new camera matrix and undistortion map
        optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (frame_width, frame_height),
            1,
            (frame_width, frame_height),
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            optimal_camera_matrix,
            (frame_width, frame_height),
            cv2.CV_16SC2,
        )

        for _ in tqdm(range(frame_count), desc="Undistorting video"):
            ret, frame = self.read()
            if not ret:
                break

            undistorted_frame = cv2.remap(
                frame, map1, map2, interpolation=cv2.INTER_LINEAR
            )
            out.write(undistorted_frame)

        out.release()
        print(f"Undistorted video saved to {output_video_path}")

    def compute_intensity(self):
        """
        Generate a :class:`pyneon.Stream` object containing the
        mean intensity of each video frame.

        Returns
        -------
        pyneon.Stream
            A ``Stream`` instance containing data indexed by ``timestamp [ns]``
            with a single column ``intensity``
        """
        vals = []
        self.reset()
        for _ in tqdm(range(len(self)), desc="Computing frame intensities"):
            ret, frame = self.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            vals.append(brightness)
        vals = np.array(vals)
        assert len(vals) == len(self.ts)
        stream = Stream(pd.DataFrame({"timestamp [ns]": self.ts, "intensity": vals}))
        return stream
