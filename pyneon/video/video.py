from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Union
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from ..utils.variables import default_camera_info
from ..vis.video import (
    overlay_detections,
    overlay_scanpath,
    plot_detections,
    plot_frame,
)
from .marker import detect_markers
from .surface import detect_surface
from .utils import get_undistort_maps, resolve_processing_window


class Video(cv2.VideoCapture):
    """
    OpenCV VideoCapture wrapper that pairs a video with frame timestamps.

    Parameters
    ----------
    video_file : pathlib.Path
        Path to the video file on disk.
    timestamps : numpy.ndarray
        Frame timestamps in nanoseconds. Must match the number of frames.
    info : dict or None
        Camera metadata, typically including ``camera_matrix`` and
        ``distortion_coefficients``.

    Attributes
    ----------
    timestamps : numpy.ndarray
        Frame timestamps in nanoseconds.
    ts : numpy.ndarray
        Alias for timestamps.
    """

    def __init__(
        self, video_file: Path, timestamps: np.ndarray, info: Optional[dict] = None
    ):
        super().__init__(str(video_file))
        self.video_file = video_file
        if not self.isOpened():
            raise IOError(f"Failed to open video file: {video_file}")

        timestamps = np.asarray(timestamps, dtype=np.int64)
        self.timestamps = timestamps
        self.ts = self.timestamps
        self.info = info
        self._undistort_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._closed = False

        self.der_dir = video_file.parent / "derivatives"

        if info == {}:
            warn("Video info is empty and will be loaded from default values.")
            self.info = default_camera_info
        elif info is None:  # Eye video
            self.info = {}

        if len(self.timestamps) != self.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError(
                f"Number of timestamps ({len(self.timestamps)}) does not match "
                f"number of frames ({self.get(cv2.CAP_PROP_FRAME_COUNT)})"
            )

    def __len__(self) -> int:
        return int(len(self.ts))

    def __repr__(self) -> str:
        return f"""Video name: {self.video_file.name}
Video height: {self.height} px
Video width: {self.width} px
Number of frames: {len(self)}
First timestamp: {self.first_ts}
Last timestamp: {self.last_ts}
Duration: {self.duration:.2f} seconds
Effective FPS: {self.fps:.2f}
"""

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

    @cached_property
    def undistort_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached undistortion maps and the optimal new camera matrix."""
        return get_undistort_maps(self)

    @cached_property
    def map1(self) -> np.ndarray:
        """First undistortion map for use with ``cv2.remap``."""
        return self.undistort_cache[0]

    @cached_property
    def map2(self) -> np.ndarray:
        """Second undistortion map for use with ``cv2.remap``."""
        return self.undistort_cache[1]

    @cached_property
    def undistortion_matrix(self) -> np.ndarray:
        """Optimal new camera matrix used for undistortion."""
        return self.undistort_cache[2]

    @property
    def current_frame_index(self) -> int:
        """Current frame index based on the video position."""
        return int(self.get(cv2.CAP_PROP_POS_FRAMES))

    def timestamp_to_frame_index(
        self, timestamp: Union[int, np.int64, np.ndarray]
    ) -> np.ndarray:
        """
        Map one or many timestamps (ns) to nearest frame index/indices.

        Parameters
        ----------
        timestamp : int or numpy.ndarray
            Timestamp(s) in nanoseconds.

        Returns
        -------
        numpy.ndarray
            1D array of frame indices corresponding to the timestamp(s).

        Raises
        ------
        ValueError
            If any timestamp is earlier than the first video timestamp or
            later than the last video timestamp.
        """
        ts_array = np.atleast_1d(np.asarray(timestamp, dtype=np.int64))

        if ts_array.size > 0 and ts_array.min() < self.ts[0]:
            raise ValueError("Timestamp is earlier than the first video timestamp.")
        if ts_array.size > 0 and ts_array.max() > self.ts[-1]:
            raise ValueError("Timestamp is later than the last video timestamp.")

        frame_indices = np.searchsorted(self.ts, ts_array).astype(int)
        return frame_indices

    def read_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Read a single frame using sequential access.
        Advances by grabbing intermediate frames for forward jumps to keep
        timestamps aligned in VFR videos, rewinding to start if .

        Parameters
        ----------
        frame_index : int
            Zero-based frame index to read.

        Returns
        -------
        numpy.ndarray or None
            Frame as a 3D array (BGR), or ``None`` if the frame cannot be read.

        Raises
        ------
        ValueError
            If ``frame_index`` is out of bounds.
        """
        if frame_index < 0 or frame_index >= len(self.ts):
            raise ValueError(f"frame_index {frame_index} is out of bounds.")

        current = self.current_frame_index

        # Use grab to advance frame-by-frame when seeking forward to maintain timestamp alignment in VFR videos.
        if frame_index > current:
            while frame_index > current:
                if not self.grab():
                    warn(
                        f"Failed to grab frame while seeking forward to frame {frame_index}."
                    )
                    return None
                current += 1
        elif current != frame_index:  # Target frame is in the past
            # Reset to start and grab forward
            self.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.read_frame_at(frame_index)

        ret, frame = self.retrieve()
        if not ret or frame is None:
            # Fallback for backends where retrieve() can fail intermittently.
            # Reopen and walk sequentially to the target frame, then decode.
            self.reset()
            current = 0
            while frame_index > current:
                if not self.grab():
                    warn(
                        f"Failed to grab frame while retrying access to frame {frame_index}."
                    )
                    return None
                current += 1

            ret, frame = self.retrieve()
            if not ret or frame is None:
                # Some backends only return an image through read().
                ret, frame = self.read()
            if not ret or frame is None:
                warn(f"Failed to retrieve frame at index {frame_index}.")
                return None
        return frame

    def reset(self):
        """Reopen the video file and reset the read position to the first frame."""
        self.close()
        self.open(str(self.video_file))
        self._closed = False
        if not self.isOpened():
            raise IOError(f"Failed to reopen video file: {self.video_file}")

    def close(self) -> None:
        """Release the underlying video handle."""
        if self._closed:
            return
        if self.isOpened():
            super().release()
        self._closed = True

    def __enter__(self) -> "Video":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

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
        processing_window: Optional[tuple[int | float, int | float]] = None,
        processing_window_unit: Literal["frame", "time", "timestamp"] = "frame",
        detector_parameters: Optional[cv2.aruco.DetectorParameters] = None,
        undistort: bool = False,
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
            processing_window=processing_window,
            processing_window_unit=processing_window_unit,
            detector_parameters=detector_parameters,
            undistort=undistort,
        )

    @fill_doc
    def detect_surface(
        self,
        step: int = 1,
        processing_window: tuple[int | float, int | float] | None = None,
        processing_window_unit: Literal["frame", "time", "timestamp"] = "frame",
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.98,
        brightness_threshold: int = 180,
        adaptive: bool = True,
        morph_kernel: int = 5,
        decimate: float = 1.0,
        mode: str = "largest",
        report_diagnostics: bool = False,
        undistort: bool = False,
    ) -> Stream:
        """
        Detect bright rectangular regions (e.g., projected surfaces or monitors)
        in video frames using luminance-based contour detection.

        Parameters
        ----------
        %(detect_surface_params)s

        Returns
        -------
        %(detect_surface_return)s
        """
        return detect_surface(
            self,
            step=step,
            processing_window=processing_window,
            processing_window_unit=processing_window_unit,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            brightness_threshold=brightness_threshold,
            adaptive=adaptive,
            morph_kernel=morph_kernel,
            decimate=decimate,
            mode=mode,
            report_diagnostics=report_diagnostics,
            undistort=undistort,
        )

    @fill_doc
    def plot_detections(
        self,
        detections: Stream,
        frame_index: int,
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
        output_path: Path | str = None,
    ) -> None:
        """
        Overlay scanpath fixations on the video frames.

        The resulting video can be displayed and/or saved.

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
        output_path : pathlib.Path or str or None
            Path to save the video with fixations overlaid. If None, the video is not saved.
            If "default", saves scanpath.mp4 to the derivatives folder under the
            recording directory.

        Returns
        -------
        None
        """
        overlay_scanpath(
            self,
            scanpath,
            circle_radius,
            line_thickness,
            max_fixations,
            show_video,
            output_path,
        )

    @fill_doc
    def overlay_detections(
        self,
        detections: "Stream",
        show_ids: bool = True,
        color: tuple[int, int, int] = (255, 0, 255),
        show_video: bool = False,
        output_path: Optional[Path | str] = None,
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
        %(output_path_param)s
            If "default", saves detections.mp4 to the derivatives folder under the
            recording directory. If None, no output video is written.

        Returns
        -------
        None
        """
        overlay_detections(
            self,
            detections=detections,
            show_ids=show_ids,
            color=color,
            show_video=show_video,
            output_path=output_path,
        )

    def undistort_video(
        self,
        show: bool = False,
        output_path: Optional[Path | str] = None,
    ) -> None:
        """
        Undistort a video using the known camera matrix and distortion coefficients.

        Parameters
        ----------
        output_path : pathlib.Path or str or None
            Path to save the undistorted output video.
            If "default", saves undistorted_video.mp4 to the derivatives folder under
            the recording directory. If None, no output video is written.

        Returns
        -------
        None
        """
        if output_path is None and not show:
            raise ValueError("Either show=True or output_path must be provided.")
        if output_path == "default":
            output_path = (
                self.video_file.parent / "derivatives" / "undistorted_video.mp4"
            )

        # Open the input video
        self.reset()
        # Get self properties
        frame_width, frame_height = self.width, self.height
        fps = self.fps
        frame_count = len(self)

        # Prepare output video writer
        out = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Adjust codec as needed
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (frame_width, frame_height)
            )

        for frame_index in tqdm(range(frame_count), desc="Undistorting video"):
            frame = self.read_frame_at(frame_index)
            if frame is None:
                break
            undistorted = self.undistort_frame(frame)

            if show:
                cv2.namedWindow("Undistorted Video", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    "Undistorted Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
                cv2.imshow("Undistorted Video", undistorted)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if out is not None:
                out.write(undistorted)

        if out is not None:
            out.release()
        if show:
            cv2.destroyAllWindows()
        self.reset()

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a single frame (color or grayscale).

        Parameters
        ----------
        frame : numpy.ndarray
            Input frame as a 2D (grayscale) or 3D (color) array.

        Returns
        -------
        numpy.ndarray
            Undistorted frame with the same shape as the input.
        """
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    @fill_doc
    def compute_frame_brightness(
        self,
        step: int = 1,
        processing_window: Optional[tuple[int | float, int | float]] = None,
        processing_window_unit: Literal["frame", "time", "timestamp"] = "frame",
    ):
        """
        Compute per-frame mean grayscale brightness.

        Each frame is converted to grayscale and averaged to yield a single
        brightness value per frame.

        Parameters
        ----------
        %(step_param)s
        %(window_params)s

        Returns
        -------
        Stream
            Stream indexed by ``timestamp [ns]`` with a single column
            ``brightness`` containing mean grayscale brightness values.
        """
        # Determine frame range
        start_frame, end_frame = resolve_processing_window(
            self,
            processing_window,
            processing_window_unit,
        )

        vals = []
        frame_indices = []

        for frame_index in tqdm(
            range(start_frame, end_frame + 1, step),
            desc="Computing frame brightness",
        ):
            frame = self.read_frame_at(frame_index)
            if frame is None:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            vals.append(brightness)
            frame_indices.append(frame_index)

        vals = np.array(vals)
        stream = Stream(
            pd.DataFrame(
                {
                    "timestamp [ns]": self.ts[frame_indices],
                    "brightness": vals,
                }
            )
        )
        return stream
