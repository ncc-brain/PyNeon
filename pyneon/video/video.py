import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from ..vis import plot_frame, overlay_scanpath
from .apriltag import detect_apriltags


class SceneVideo(cv2.VideoCapture):
    """
    Loaded video file with timestamps.

    Parameters
    ----------
    video_file : pathlib.Path
        Path to the video file.
    timestamps_file : pathlib.Path
        Path to the timestamps file.
    info_file : pathlib.Path
        Path to the scene camera info file.

    Attributes
    ----------
    timestamps : numpy.ndarray
        Timestamps of the video frames in nanoseconds.
    ts : numpy.ndarray
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
        print("Resetting video...")
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

    def detect_apriltags(self, tag_family: str = "tag36h11") -> pd.DataFrame:
        """
        Detect AprilTags in the video frames.

        Parameters
        ----------
        tag_family : str, optional
            The AprilTag family to detect (default is 'tag36h11').

        Returns
        -------
        pd.DataFrame
            A DataFrame containing AprilTag detections, with columns:
            - 'timestamp [ns]': The timestamp of the frame in nanoseconds, as an index
            - 'frame_idx': The frame number
            - 'tag_id': The ID of the detected AprilTag
            - 'corners': A 4x2 array of the tag corner coordinates, in the order TL, TR, BR, BL. (x, y) from top-left corner of the video
            - 'center': A 1x2 array with the tag center coordinates. (x, y) from top-left corner of the video.
        """

        return detect_apriltags(self, tag_family)

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
        dist_coeffs = self.dist_coeffs

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

        for frame_idx in tqdm(range(frame_count), desc="Undistorting video"):
            ret, frame = self.read()
            if not ret:
                break

            undistorted_frame = cv2.remap(
                frame, map1, map2, interpolation=cv2.INTER_LINEAR
            )
            out.write(undistorted_frame)

        out.release()
        print(f"Undistorted video saved to {output_video_path}")
