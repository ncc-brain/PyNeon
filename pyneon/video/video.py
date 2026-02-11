from pathlib import Path
from typing import Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from ..utils.variables import default_camera_info
from ..vis import (
    overlay_scanpath,
    plot_detections,
    plot_frame,
    overlay_detections,
)
from .marker import detect_markers
from .surface import detect_surface


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
        self._undistort_maps: tuple[np.ndarray, np.ndarray] | None = None

        self.der_dir = video_file.parent / "derivatives"

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
        """First timestamp of the video."""
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Last timestamp of the video."""
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
        """Duration of the video in seconds."""
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

    @property
    def map1(self) -> np.ndarray:
        return self._get_undistort_maps()[0]

    @property
    def map2(self) -> np.ndarray:
        return self._get_undistort_maps()[1]

    def timestamp_to_frame_index(
        self, timestamp: Union[int, np.int64, np.ndarray]
    ) -> np.ndarray:
        """
        Map one or many timestamps (ns) to the corresponding frame index/indices.

        Parameters
        ----------
        timestamp : int or numpy.ndarray
            Timestamp(s) in nanoseconds.

        Returns
        -------
        numpy.ndarray
            Frame index/indices corresponding to the timestamp(s).
        """
        ts_array = np.atleast_1d(np.asarray(timestamp, dtype=np.int64))

        if ts_array.size > 0 and (
            ts_array.min() < self.ts[0] or ts_array.max() > self.ts[-1]
        ):
            raise ValueError("Timestamp is out of bounds of the video timestamps.")

        indices = np.searchsorted(self.ts, ts_array).astype(int)
        return indices

    def read_gray_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Random-access read of a single frame converted to grayscale.
        Uses sequential grabbing for short forward jumps to maintain frame accuracy
        in VFR videos, and fallback to seeking for large jumps.

        Returns None if the frame cannot be read.
        """
        if frame_index < 0 or frame_index >= len(self.ts):
            raise ValueError(f"frame_index {frame_index} is out of bounds.")

        current = int(self.get(cv2.CAP_PROP_POS_FRAMES))

        # If we are slightly behind (up to 50 frames), use grab() to stay accurate.
        # This is much faster than decoding and avoids seeker drift in VFR/MSMF.
        if 0 <= (frame_index - current) < 50:
            while current < frame_index:
                if not self.grab():
                    return None
                current += 1
        elif current != frame_index:
            self.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = self.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def reset(self):
        if self.isOpened():
            self.release()
        super().__init__(self.video_file)
        self.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not self.isOpened():
            raise IOError(f"Failed to reopen video file: {self.video_file}")

    @fill_doc
    def plot_frame(
        self,
        frame_index: int = 0,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """
        Plot a frame from the video on a matplotlib axis.

        Parameters
        ----------
        frame_index : int
            Index of the frame to plot.
        %(ax_param)s

        Returns
        -------
        %(fig_ax_return)s
        """
        return plot_frame(self, frame_index, ax, show)

    @fill_doc
    def detect_markers(
        self,
        marker_family: str | list[str] = "36h11",
        step: int = 1,
        detection_window: Optional[tuple[int | float, int | float]] = None,
        detection_window_unit: str = "frame",
        detector_parameters: Optional[cv2.aruco.DetectorParameters] = None,
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
            marker_family=marker_family,
            step=step,
            detection_window=detection_window,
            detection_window_unit=detection_window_unit,
            detector_parameters=detector_parameters,
        )

    def detect_surface(
        self,
        skip_frames: int = 1,
        detection_window: tuple[int | float, int | float] | None = None,
        detection_window_unit: str = "frame",
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.98,
        brightness_threshold: int = 180,
        adaptive: bool = True,
        morph_kernel: int = 5,
        decimate: float = 1.0,
        mode: str = "largest",
        report_diagnostics: bool = False,
    ) -> Stream:
        """
        Detect bright rectangular regions (e.g., projected screens or monitors)
        in video frames using luminance-based contour detection.

        Parameters
        ----------
        skip_frames : int, optional
            Process every Nth frame (default 1 = process all frames).
        detection_window : tuple, optional
            A tuple (start, end) specifying the range to search for detections.
            Interpretation depends on `detection_window_unit`. Defaults to ``None``.
        detection_window_unit : {"frame", "time", "timestamp"}, optional
            Unit for values in `detection_window`. Defaults to "frame".
        min_area_ratio : float, optional
            Minimum contour area relative to frame area. Contours smaller than this
            ratio are ignored. Default is 0.01 (1% of frame area).
        max_area_ratio : float, optional
            Maximum contour area relative to frame area. Contours larger than this
            ratio are ignored. Default is 0.98.
        brightness_threshold : int, optional
            Fixed threshold for binarization when `adaptive=False`. Default is 180.
        adaptive : bool, optional
            If True (default), use adaptive thresholding to handle varying
            illumination across frames.
        morph_kernel : int, optional
            Kernel size for morphological closing (default 5). Use 0 to disable
            morphological operations.
        decimate : float, optional
            Downsampling factor for faster processing (e.g., 0.5 halves resolution).
            Detected coordinates are automatically rescaled back. Default is 1.0.
        mode : {"largest", "best", "all"}, optional
            Selection mode determining which contours to return per frame.
        report_diagnostics : bool, optional
            If True, includes "area_ratio" and "score" columns in the output.

        Returns
        -------
        Stream
            One row per detected rectangular contour.
        """
        return detect_surface(
            self,
            skip_frames=skip_frames,
            detection_window=detection_window,
            detection_window_unit=detection_window_unit,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            brightness_threshold=brightness_threshold,
            adaptive=adaptive,
            morph_kernel=morph_kernel,
            decimate=decimate,
            mode=mode,
            report_diagnostics=report_diagnostics,
        )

    @fill_doc
    def plot_detections(
        self,
        detections: Stream,
        frame_index: int = 0,
        show_ids: bool = True,
        color: str = "magenta",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ):
        """
        Plot detections on a frame from this video.

        Parameters
        ----------
        detections : Stream
            Stream containing marker or surface-corner detections.
        frame_index : int
            Index of the frame to plot.
        show_ids : bool
            Display detection IDs at their centers when available. Defaults to True.
        color : str
            Matplotlib color for detections. Defaults to "magenta".
        %(ax_param)s
        %(show_param)s

        Returns
        -------
        %(fig_ax_return)s
        """
        return plot_detections(
            self,
            detections=detections,
            frame_index=frame_index,
            show_ids=show_ids,
            color=color,
            ax=ax,
            show=show,
        )

    def overlay_scanpath(
        self,
        scanpath: pd.DataFrame,
        circle_radius: int = 10,
        line_thickness: int = 2,
        max_fixations: int = 10,
        show_video: bool = False,
        video_output_path: Path | str = None,
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
            Defaults to 'derivatives/scanpath.mp4'.
        """
        if video_output_path is None:
            video_output_path = self.der_dir / "scanpath.mp4"
            os.makedirs(self.der_dir, exist_ok=True)

        overlay_scanpath(
            self,
            scanpath,
            circle_radius,
            line_thickness,
            max_fixations,
            show_video,
            video_output_path,
        )

    @fill_doc
    def overlay_detections(
        self,
        detections: "Stream",
        show_ids: bool = True,
        color: tuple[int, int, int] = (255, 0, 255),
        show_video: bool = False,
        video_output_path: Optional[Path | str] = None,
    ) -> None:
        """
        Overlay detections on the video frames.
        The resulting video can be displayed and/or saved.

        Parameters
        ----------
        detections : Stream
            Stream containing marker or surface-corner detections.
        show_ids : bool
            Whether to overlay IDs at their centers when available. Defaults to True.
        color : tuple[int, int, int]
            BGR color tuple for overlays. Defaults to (255, 0, 255) which is magenta.
        %(show_video_param)s
        %(video_output_path_param)s
            Defaults to 'derivatives/detected_markers.mp4'.
        """
        if video_output_path is None:
            video_output_path = self.der_dir / "detected_markers.mp4"
            os.makedirs(self.der_dir, exist_ok=True)

        overlay_detections(
            self,
            detections=detections,
            show_ids=show_ids,
            color=color,
            show_video=show_video,
            video_output_path=video_output_path,
        )

    def undistort_video(
        self,
        output_video_path: Optional[Path | str] = None,
    ) -> None:
        """
        Undistort a video using the known camera matrix and distortion coefficients.

        Parameters
        ----------
        output_video_path : str
            Path to save the undistorted output video. Defaults to 'undistorted_video.mp4'.
        """
        if output_video_path is None:
            output_video_path = self.der_dir / "undistorted_video.mp4"
            os.makedirs(self.der_dir, exist_ok=True)

        # Open the input video
        cap = self
        cap.reset()
        # Get self properties
        frame_width, frame_height = self.width, self.height
        fps = self.fps
        frame_count = len(self)

        # Prepare output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Adjust codec as needed
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

        for _ in tqdm(range(frame_count), desc="Undistorting video"):
            ret, frame = self.read()
            if not ret:
                break

            out.write(self.undistort_frame(frame))

        out.release()
        print(f"Undistorted video saved to {output_video_path}")

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort a single frame (color or grayscale)."""
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def _get_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        if self._undistort_maps is None:
            camera_matrix = self.camera_matrix
            dist_coeffs = self.distortion_coefficients
            frame_width, frame_height = self.width, self.height
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
            self._undistort_maps = (map1, map2)
        return self._undistort_maps

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
