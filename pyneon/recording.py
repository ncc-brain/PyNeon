import json
import shutil
from datetime import datetime
from functools import cached_property
from numbers import Number
from pathlib import Path
from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .events import Events
from .export import export_cloud_format, export_eye_bids, export_motion_bids
from .preprocess import concat_events, concat_streams, smooth_camera_pose
from .stream import Stream
from .utils.doc_decorators import fill_doc
from .utils.variables import calib_dtype, expected_files_cloud, expected_files_native
from .video import (
    Video,
    detect_markers,
    estimate_camera_pose,
    estimate_scanpath,
    find_homographies,
    gaze_on_surface,
)
from .vis import overlay_detections_and_pose, overlay_scanpath, plot_distribution


class Recording:
    """
    Container of a multi-modal recording with streams, events, and videos.

    The recording directory is expected to follow either the Pupil Cloud format
    (tested with data format version >= 2.3) or the native Pupil Labs format.
    (tested with data format version >= 2.5).
    In both cases, the directory must contain an ``info.json`` file.

    Example Pupil Cloud recording directory structure:

    .. code-block:: text

        recording_dir/
        ├── 3d_eye_states.csv
        ├── blinks.csv
        ├── events.csv
        ├── fixations.csv
        ├── gaze.csv
        ├── imu.csv
        ├── saccades.csv
        ├── info.json (REQUIRED)
        ├── labels.csv
        ├── saccades.csv
        ├── scene_camera.json
        ├── world_timestamps.csv
        └── *.mp4

    Example native Pupil Labs recording directory structure:

    .. code-block:: text

        recording_dir/
        ├── blinks ps1.raw
        ├── blinks ps1.time
        ├── blinks.dtype
        ├── calibration.bin
        ├── event.time
        ├── event.txt
        ├── ...
        ├── gaze ps1.raw
        ├── gaze ps1.time
        ├── gaze.dtype
        ├── ...
        ├── info.json (REQUIRED)
        ├── Neon Scene Camera v1 ps1.mp4
        ├── Neon Scene Camera v1 ps1.time
        ├── Neon Sensor Module v1 ps1.mp4
        ├── Neon Sensor Module v1 ps1.time
        ├── ...
        ├── wearer.json
        ├── worn ps1.raw
        └── worn.dtype

    Streams, events, and scene video will be located but not loaded until
    accessed as properties such as ``gaze``, ``imu``, and ``eye_states``.

    Parameters
    ----------
    recording_dir : str or pathlib.Path
        Path to the directory containing the recording.

    Attributes
    ----------
    recording_id : str
        Recording ID.
    recording_dir : pathlib.Path
        Path to the recording directory.
    format : {'cloud', 'native'}
        Recording format, either "cloud" for Pupil Cloud format or "native" for
        native format.
    info : dict
        Information about the recording. Read from ``info.json``. For details, see
        https://docs.pupil-labs.com/neon/data-collection/data-format/#info-json.
    data_format_version : str | None
        Data format version as in ``info.json``.
    """

    def __init__(self, recording_dir: str | Path):
        recording_dir = Path(recording_dir)
        if not recording_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {recording_dir}")
        if not (info_path := recording_dir / "info.json").is_file():
            raise FileNotFoundError(f"info.json not found in {recording_dir}")

        with open(info_path) as f:
            info = json.load(f)

        self.recording_id = info["recording_id"]
        self.recording_dir = recording_dir

        self._infer_format()
        self._warn_missing()

        if self.format == "native":
            try:
                with open(recording_dir / "wearer.json") as wearer_path:
                    wearer_info = json.load(wearer_path)
                info["wearer_name"] = wearer_info.get("name", None)
            except FileNotFoundError:
                info["wearer_name"] = None
                warn("Cannot find wearer.json in native recording.")
        self.info = info
        self.data_format_version = info.get("data_format_version", None)

        self.der_dir = recording_dir / "derivatives"
        if not self.der_dir.is_dir():
            self.der_dir.mkdir()

    def _infer_format(self):
        gaze_csv = self.recording_dir / "gaze.csv"
        gaze_ps1_time = self.recording_dir / "gaze ps1.time"
        gaze_ps1_raw = self.recording_dir / "gaze ps1.raw"

        if gaze_csv.is_file():
            self.format = "cloud"
        elif gaze_ps1_time.is_file() and gaze_ps1_raw.is_file():
            self.format = "native"
        else:
            raise FileNotFoundError(
                f"Cannot infer recording type in directory: {self.recording_dir}"
            )

    def _warn_missing(self):
        if self.format == "cloud":
            exp_files = expected_files_cloud
        elif self.format == "native":
            exp_files = expected_files_native

        files = list(self.recording_dir.glob("*"))
        missing_files = [f for f in exp_files if (self.recording_dir / f) not in files]

        if self.format == "cloud" and (not list(self.recording_dir.glob("*.mp4"))):
            missing_files.append("*.mp4")
        if missing_files:
            missing_files = "\n".join(missing_files)
            warn(
                f"Recording {self.recording_id} misses the following expected files:\n{missing_files}",
                UserWarning,
            )

    def __repr__(self) -> str:
        return f"""
Data format: {self.format}
Recording ID: {self.recording_id}
Wearer ID: {self.info.get("wearer_id", "N/A")}
Wearer name: {self.info.get("wearer_name", "N/A")}
Recording start time: {self.start_datetime}
Recording duration: {self.info["duration"]} ns ({self.info["duration"] / 1e9} s)
"""

    @cached_property
    def gaze(self) -> Stream:
        """
        Return a cached :class:`pyneon.Stream` object containing gaze data.

        For **Pupil Cloud** recordings, the data is loaded from ``gaze.csv``.

        For **native** recordings, the data is loaded from ``gaze_200hz.raw`` (if present;
        otherwise from ``gaze ps1.raw``) along with the corresponding ``.time`` and
        ``.dtype`` files.
        """
        if self.format == "native":
            gaze_200hz_file = self.recording_dir / "gaze_200hz.raw"
            if gaze_200hz_file.is_file():
                return Stream(gaze_200hz_file)
            else:
                return Stream(self.recording_dir / "gaze ps1.raw")
        else:
            return Stream(self.recording_dir / "gaze.csv")

    @cached_property
    def imu(self) -> Stream:
        """
        Return a cached :class:`pyneon.Stream` object containing IMU data.

        For **Pupil Cloud** recordings, the data is loaded from ``imu.csv``.

        For **native** recordings, the data is loaded from ``imu ps1.raw``, along with
        the corresponding ``.time`` and ``.dtype`` files.
        """
        if self.format == "native":
            return Stream(self.recording_dir / "imu ps1.raw")
        else:
            return Stream(self.recording_dir / "imu.csv")

    @cached_property
    def eye_states(self) -> Stream:
        """
        Returns a (cached) :class:`pyneon.Stream` object containing eye states data.

        For **Pupil Cloud** recordings, the data is loaded from ``3d_eye_states.csv``.

        For **native** recordings, the data is loaded from ``eye_state ps1.raw``, along
        with the corresponding ``.time`` and ``.dtype`` files.
        """
        if self.format == "native":
            return Stream(self.recording_dir / "eye_state ps1.raw")
        else:
            return Stream(self.recording_dir / "3d_eye_states.csv")

    @cached_property
    def blinks(self) -> Events:
        """
        Return a cached :class:`pyneon.Events` object containing blink event data.

        For **Pupil Cloud** recordings, the data is loaded from ``blinks.csv``.

        For **native** recordings, the data is loaded from ``blinks ps1.raw``, along with
        the corresponding ``.time`` and ``.dtype`` files.
        """
        if self.format == "native":
            return Events(self.recording_dir / "blinks ps1.raw")
        else:
            return Events(self.recording_dir / "blinks.csv")

    @cached_property
    def fixations(self) -> Events:
        """
        Returns a (cached) :class:`pyneon.Events` object containing fixations data.

        For **Pupil Cloud** recordings, the data is loaded from ``fixations.csv``.

        For **native** recordings, the data is loaded from ``fixations ps1.raw``,
        along with the corresponding ``.time`` and ``.dtype`` files.
        """
        if self.format == "native":
            return Events(self.recording_dir / "fixations ps1.raw", "fixations")
        else:
            return Events(self.recording_dir / "fixations.csv")

    @cached_property
    def saccades(self) -> Events:
        """
        Returns a (cached) :class:`pyneon.Events` object containing saccades data.

        For **Pupil Cloud** recordings, the data is loaded from ``saccades.csv``.

        For **native** recordings, the data is loaded from ``fixations ps1.raw``,
        along with the corresponding ``.time`` and ``.dtype`` files.
        """
        if self.format == "native":
            # Note: In the native format, both fixations and saccades are stored in 'fixations ps1.raw'.
            # The event type argument ("saccades") is used to distinguish which events to load.
            return Events(self.recording_dir / "fixations ps1.raw", "saccades")
        else:
            return Events(self.recording_dir / "saccades.csv")

    @cached_property
    def events(self) -> Events:
        """
        Returns a (cached) :class:`pyneon.Events` object containing events data.

        For **Pupil Cloud** recordings, the events data is loaded from ``events.csv``.

        For **native** recordings, the events data is loaded from ``event.txt`` and
        ``event.time``.
        """
        if self.format == "native":
            return Events(self.recording_dir / "event.txt")
        else:
            return Events(self.recording_dir / "events.csv")

    @cached_property
    def scene_video(self) -> Video:
        """
        Returns a (cached) :class:`pyneon.Video` object containing scene video data.

        For **Pupil Cloud** recordings, the video is loaded from the only ``*.mp4`` file
        in the recording directory.

        For **native** recordings, the video is loaded from the ``Neon Scene Camera*.mp4``
        file in the recording directory.
        """
        reg_exp = "Neon Scene Camera*.mp4" if self.format == "native" else "*.mp4"
        video_files = list(self.recording_dir.glob(reg_exp))
        if len(video_files) == 0:
            raise FileNotFoundError(
                f"No scene video file found in {self.recording_dir}"
            )
        elif len(video_files) > 1:
            raise FileNotFoundError(
                f"Multiple scene video files found in {self.recording_dir}"
            )
        video_file = video_files[0]
        if self.format == "native":
            ts_file = video_file.with_suffix(".time")
            if not ts_file.is_file():
                raise FileNotFoundError(
                    f"Missing {ts_file.name} in {self.recording_dir}"
                )
            ts = np.fromfile(ts_file, dtype=np.int64)
            calib_file = self.recording_dir / "calibration.bin"
            if not calib_file.is_file():
                warn(
                    f"Missing calibration.bin in {self.recording_dir}; "
                    "scene camera info will be empty."
                )
                info = {}
            else:
                dtype = np.dtype(calib_dtype)
            calibration = np.frombuffer(calib_file.open("rb").read(), dtype)[0]
            info = {
                "camera_matrix": calibration["scene_camera_matrix"].tolist(),
                "distortion_coefficients": calibration[
                    "scene_distortion_coefficients"
                ].tolist(),
                "serial_number": calibration["serial"].decode("utf-8"),
            }
        else:
            ts_file = self.recording_dir / "world_timestamps.csv"
            info_file = self.recording_dir / "scene_camera.json"
            if not ts_file.is_file():
                raise FileNotFoundError(
                    f"Missing world_timestamps.csv in {self.recording_dir}"
                )
            ts = pd.read_csv(ts_file)["timestamp [ns]"].to_numpy().astype(np.int64)
            if not info_file.is_file():
                raise FileNotFoundError(
                    f"Missing scene_camera.json in {self.recording_dir}"
                )
            with open(info_file) as f:
                info = json.load(f)
        return Video(video_file, ts, info)

    @cached_property
    def eye_video(self) -> Video:
        """
        Returns a (cached) :class:`pyneon.Video` object containing eye video data.

        Eye video is only available for **native** recordings and is loaded from the
        ``Neon Sensor Module*.mp4`` file in the recording directory.
        """
        if self.format == "cloud":
            raise ValueError("Pupil Cloud recordings do not contain eye video.")
        video_files = list(self.recording_dir.glob("Neon Sensor Module*.mp4"))
        if len(video_files) == 0:
            raise FileNotFoundError(f"No eye video file found in {self.recording_dir}")
        elif len(video_files) > 1:
            raise FileNotFoundError(
                f"Multiple eye video files found in {self.recording_dir}"
            )
        video_file = video_files[0]
        ts_file = video_file.with_suffix(".time")
        if not ts_file.is_file():
            raise FileNotFoundError(
                f"Missing .time file {ts_file.name} in {self.recording_dir}"
            )
        ts = np.fromfile(ts_file, dtype=np.int64)
        return Video(video_file, ts, {})

    @property
    def start_time(self) -> int:
        """
        Start time (in ns) of the recording as in ``info.json``.
        May not match the start time of each data stream.
        """
        return self.info["start_time"]

    @property
    def start_datetime(self) -> datetime:
        """
        Start time (datetime) of the recording as in ``info.json``.
        May not match the start time of each data stream.
        """
        return datetime.fromtimestamp(self.start_time / 1e9)

    def concat_streams(
        self,
        stream_names: str | list[str],
        sampling_freq: Number | str = "min",
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
        inplace: bool = False,
    ) -> Stream:
        """
        Concatenate data from different streams under common timestamps.
        Since the streams may have different timestamps and sampling frequencies,
        resampling of all streams to a set of common timestamps is performed.
        The latest start timestamp and earliest last timestamp of the selected sreams
        are used to define the common timestamps.

        Parameters
        ----------
        stream_names : str or list of str
            Stream names to concatenate. If "all", then all streams will be used.
            If a list, items must be in ``{"gaze", "imu", "eye_states"}``
            ("3d_eye_states" is also tolerated as an alias for "eye_states").
        sampling_freq : float or int or str, optional
            Sampling frequency of the concatenated streams.
            If numeric, the streams will be interpolated to this frequency.
            If "min" (default), the lowest nominal sampling frequency
            of the selected streams will be used.
            If "max", the highest nominal sampling frequency will be used.
        float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            Defaults to "linear". For details see :class:`scipy.interpolate.interp1d`.
        other_kind : str, optional
            Kind of interpolation applied on columns of other types,
            Defaults to "nearest". Only "nearest", "previous", and "next" are recommended.
        inplace : bool, optional
            Replace selected stream data with interpolated data during concatenation
            if ``True``. Defaults to ``False``.

        Returns
        -------
        Stream
            Stream object containing concatenated data.
        """
        return Stream(
            concat_streams(
                self,
                stream_names,
                sampling_freq,
                float_kind,
                other_kind,
                inplace,
            )
        )

    def concat_events(self, event_names: str | list[str]) -> Events:
        """
        Concatenate different events. All columns in the selected event type will be
        present in the final DataFrame. An additional "type" column denotes the event
        type. If ``events`` is selected, its "timestamp [ns]" column will be
        renamed to "start timestamp [ns]", and the "name" and "type" columns will
        be renamed to "message name" and "message type" respectively to provide
        a more readable output.

        Parameters
        ----------
        event_names : list of str
            List of event names to concatenate. Event names must be in
            ``{"blinks", "fixations", "saccades", "events"}``
            (singular forms are tolerated).

        Returns
        -------
        Events
            Events object containing concatenated data.
        """
        new_data = concat_events(self, event_names)
        return Events(new_data)

    def plot_distribution(
        self,
        heatmap_source: Literal["gaze", "fixations", None] = "gaze",
        scatter_source: Literal["gaze", "fixations", None] = "fixations",
        show: bool = True,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a heatmap of gaze or fixation data on a matplotlib axis.
        Users can flexibly choose to generate a smoothed heatmap and/or scatter plot and
        the source of the data (gaze or fixation).

        Parameters
        ----------
        rec : Recording
            Recording object containing the gaze and video data.
        heatmap_source : {'gaze', 'fixations', None}
            Source of the data to plot as a heatmap. If None, no heatmap is plotted.
            Defaults to 'gaze'.
        scatter_source : {'gaze', 'fixations', None}
            Source of the data to plot as a scatter plot. If None, no scatter plot is plotted.
            Defaults to 'fixations'. Gaze data is typically more dense and thus less suitable
            for scatter plots.
        show : bool
            Show the figure if ``True``. Defaults to True.
        **kwargs : keyword arguments
            Additional parameters for the plot, including:
                - 'step_size': Step size for the heatmap grid. Default is 10.
                - 'sigma': Sigma value for Gaussian smoothing. Default is 2.
                - 'width_height': Width and height of the figure in pixels. Default is (1600, 1200).
                - 'cmap': Colormap for the heatmap. Default is 'inferno'.
                - 'ax': Matplotlib axis to plot on. If None, a new figure and axis are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.
        ax : matplotlib.axes.Axes
            Axis object containing the plot.
        """
        step_size = kwargs.get("step_size", 10)
        sigma = kwargs.get("sigma", 2)
        width_height = kwargs.get("width_height", (1600, 1200))
        cmap = kwargs.get("cmap", "inferno")
        ax = kwargs.get("ax", None)

        return plot_distribution(
            self,
            heatmap_source,
            scatter_source,
            step_size,
            sigma,
            width_height,
            cmap,
            ax,
            show,
        )

    def sync_gaze_to_video(
        self,
        window_size: Optional[int] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Synchronize gaze data to video frames by applying windowed averaging
        around timestamps of each video frame.

        Parameters
        ----------
        window_size : int, optional
            Size of the time window in nanoseconds used for averaging.
            If None, defaults to the median interval between video frame timestamps.
        overwrite : bool, optional
            If True, force recomputation even if saved data exists. Default is False.
        output_path : str or pathlib.Path, optional
            Path to save the resulting CSV file. Defaults to `<der_dir>/gaze_synced.csv`.

        Returns
        -------
        Stream
            A Stream indexed by `"timestamp [ns]"`, containing:
                - 'gaze x [px]': Gaze x-coordinate in pixels
                - 'gaze y [px]': Gaze y-coordinate in pixels
                - 'frame id': Index of the video frame corresponding to the gaze data
        """
        if output_path is None:
            gaze_file = self.der_dir / "gaze_synced.csv"
        else:
            gaze_file = Path(output_path)

        if gaze_file.is_file() and not overwrite:
            synced_gaze = pd.read_csv(gaze_file, index_col="timestamp [ns]")
            if synced_gaze.empty:
                raise ValueError("Gaze data is empty.")
            return Stream(synced_gaze)

        if self.gaze is None or self.scene_video is None:
            raise ValueError("Gaze-video synchronization requires gaze and video data.")

        synced_gaze = self.gaze.window_average(self.scene_video.ts, window_size).data
        synced_gaze["frame id"] = np.arange(len(synced_gaze))

        synced_gaze.index.name = "timestamp [ns]"
        synced_gaze.to_csv(gaze_file, index=True)

        return Stream(synced_gaze)

    def estimate_scanpath(
        self,
        sync_gaze: Optional[Stream] = None,
        lk_params: Optional[dict] = None,
        output_path: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Stream:
        """
        Estimate scanpaths by propagating fixations across frames with Lucas-Kanade.

        If a cached result exists and ``overwrite`` is False, it is loaded from disk.

        Parameters
        ----------
        sync_gaze : Stream, optional
            Gaze data synchronised to video frames. If ``None``, it is created with
            Recording.sync_gaze_to_video().
        lk_params : dict, optional
            Parameters forwarded to the LK optical-flow call.
        output_path : str or pathlib.Path, optional
            Where to save the pickle. Defaults to ``<der_dir>/scanpath.pkl``.
        overwrite : bool, optional
            Force recomputation even if a pickle exists.

        Returns
        -------
        Stream
            Indexed by "timestamp [ns]" with one column "fixations"
            containing a nested DataFrame for each video frame.
        """
        scanpath_path = (
            Path(output_path) if output_path else self.der_dir / "scanpath.pkl"
        )

        if scanpath_path.is_file() and not overwrite:
            print(f"Loading saved scanpath from {scanpath_path}")
            df = pd.read_pickle(scanpath_path)
            if df.empty:
                raise ValueError("Scanpath data is empty.")
            return Stream(df)

        if sync_gaze is None:
            sync_gaze = self.sync_gaze_to_video()

        scanpath_df = estimate_scanpath(
            self.scene_video,
            sync_gaze,
            lk_params=lk_params,
        )
        scanpath_df.index.name = "timestamp [ns]"

        # ------------------------------------------------------------------ save
        scanpath_df.to_pickle(scanpath_path)

        return Stream(scanpath_df)

    @fill_doc
    def detect_markers(
        self,
        marker_name: str = "tag36h11",
        step: int = 1,
        detection_window: Optional[tuple[int | float, int | float]] = None,
        detection_window_unit: str = "frame",
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Detect fiducial markers (AprilTag or ArUco) in a video and return their positions per frame.

        Uses saved results if available unless `overwrite=True`.

        Parameters
        ----------
        %(detect_markers_params)s
        overwrite : bool, optional
            If True, reruns detection even if saved results exist. Default is False.
        output_path : str or pathlib.Path, optional
            Path to save the detection CSV file. Defaults to `<der_dir>/markers.csv`.

        Returns
        -------
        %(detect_markers_return)s
        """

        window_unit = detection_window_unit
        valid_units = {"frame", "time", "timestamp"}
        if window_unit not in valid_units:
            raise ValueError(
                f"Invalid detection_window_unit '{detection_window_unit}'. Use one of {sorted(valid_units)}."
            )

        if output_path is None:
            csv_file = self.der_dir / "markers.csv"
        else:
            csv_file = Path(output_path)

        # If a saved file exists and overwrite is False, just read and return it
        if csv_file.is_file() and not overwrite:
            print(f"Loading saved detections from {csv_file}")
            all_detections = pd.read_csv(csv_file, index_col="timestamp [ns]")
            if all_detections.empty:
                raise ValueError("Marker detection data is empty.")
            return Stream(all_detections)

        all_detections = detect_markers(
            video=self.scene_video,
            marker_name=marker_name,
            step=step,
            detection_window=detection_window,
            detection_window_unit=window_unit,
        )

        if all_detections.data.empty:
            raise ValueError("No marker detections found.")

        # Save results to CSV
        all_detections.data.to_csv(csv_file, index=True)

        return all_detections

    def find_homographies(
        self,
        tag_info: pd.DataFrame,
        all_detections: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
        upsample_to: Optional[Literal["video", "gaze"]] = None,
        max_gap: Optional[int] = 5,
        extrapolate: bool = False,
        **kwargs,
    ) -> Stream:
        """
        Compute and return homographies for each frame using AprilTag detections and reference marker layout.

        Parameters
        ----------
        tag_info : pandas.DataFrame
            DataFrame containing AprilTag reference positions and orientations, with columns:
                - 'tag_id': ID of the tag
                - 'x', 'y', 'z': 3D coordinates of the tag's center
                - 'normal_x', 'normal_y', 'normal_z': Normal vector of the tag surface
                - 'size': Side length of the tag in meters
        all_detections : Stream, optional
            Stream containing AprilTag detection results per frame. If None, detections are recomputed.
        overwrite : bool, optional
            Whether to force recomputation even if saved homographies exist.
        output_path : str or pathlib.Path, optional
            Optional file path for saving the homographies as JSON. If None, defaults to `<der_dir>/homographies.json`.
        upsample_to : str, optional
            If "video", the homographies will be upsampled to match the video frames
            from the first to the last frame. If "gaze", the homographies will be
            resampled to the timestamps of the recording's gaze data. Default is None.
        max_gap : int, optional
            Maximum number of frames to interpolate across when filling gaps without
            detections. If a gap exceeds this threshold, it is filled with None homographies
            instead. Default is None (unlimited interpolation).
        extrapolate : bool, optional
            Whether to extrapolate homographies before the first detection and after
            the last detection. If False, these periods will have None homographies.
            Default is True.
        **kwargs : keyword arguments
            Additional parameters for homography computation, including:
                - 'coordinate_system': Coordinate system for the homography ('opencv' or 'psychopy'). Default is 'opencv'.
                - 'surface_size': Size of the surface in pixels (width, height). Default is (1920, 1080).
                - 'skip_frames': Number of frames to skip between detections. Default is 1.
                - 'settings': Additional settings for the homography computation.

        Returns
        -------
        Stream
            A Stream object indexed by `"timestamp [ns]"` containing:
                - 'frame_id': Video frame index
                - 'homography': 3x3 NumPy array representing the homography matrix for that frame
        """

        # Defaults for kwargs
        coordinate_system = kwargs.get("coordinate_system", "opencv")
        surface_size = kwargs.get("surface_size", (1920, 1080))
        skip_frames = kwargs.get("skip_frames", 1)
        settings = kwargs.get("settings", None)

        if output_path is None:
            pkl_file = self.der_dir / "homographies.pkl"
        else:
            pkl_file = Path(output_path)

        # If a saved file exists and overwrite is False, just read and return it
        if pkl_file.is_file() and not overwrite:
            print(f"Loading saved homographies from {pkl_file}")
            df = pd.read_pickle(pkl_file)
            homographies = Stream(df)
            return homographies

        if all_detections is None:
            all_detections = self.detect_markers()

        # if all_detections.data.empty:
        #   raise ValueError("No AprilTag detections found.")

        gaze_df = None
        if upsample_to == "gaze":
            gaze_df = self.gaze.data

        homographies_df = find_homographies(
            self.scene_video,
            all_detections.data,
            tag_info.copy(deep=True),
            surface_size,
            skip_frames=skip_frames,
            coordinate_system=coordinate_system,
            settings=settings,
            upsample_to=upsample_to,
            gaze_df=gaze_df,
            max_gap=max_gap,
            extrapolate=extrapolate,
        )

        homographies_df.to_pickle(pkl_file)

        return Stream(homographies_df)

    def gaze_on_surface(
        self,
        homographies: Optional[Stream] = None,
        tag_info: Optional[pd.DataFrame] = None,
        synced_gaze: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Project gaze coordinates from eye space to surface space using homographies.

        Computes or loads frame-wise or sample-wise homographies and applies them to
        either synchronized or raw gaze data. If homographies are already sampled to
        the raw gaze timestamps, they are applied directly. Otherwise, synchronized
        gaze is used.

        If a saved version exists and `overwrite` is False, the data is loaded from disk.

        Parameters
        ----------
        homographies : Stream, optional
            Stream containing precomputed homographies. If None, they are computed from `tag_info`.
        tag_info : pandas.DataFrame, optional
            AprilTag marker info used to compute homographies. Required if `homographies` is None.
        synced_gaze : Stream, optional
            Gaze data aligned to video frames. If None, will be either `self.gaze` or
            computed using `sync_gaze_to_video()` depending on the `homographies` frequency.
        overwrite : bool, optional
            If True, recompute and overwrite any existing surface-transformed gaze data. Default is False.
        output_path : str or pathlib.Path, optional
            File path to save the resulting CSV. Defaults to `<der_dir>/gaze_on_surface.csv`.

        Returns
        -------
        Stream
            A Stream containing gaze data with surface coordinates, including:
                - 'frame_id': Frame index
                - 'x_trans', 'y_trans': Gaze coordinates in surface pixel space
                - Any additional columns from the gaze input
        """

        if output_path is None:
            gaze_on_surface_path = self.der_dir / "gaze_on_surface.csv"
        else:
            gaze_on_surface_path = Path(output_path)

        if gaze_on_surface_path.is_file() and not overwrite:
            # Load saved gaze on surface data
            data = pd.read_csv(gaze_on_surface_path)
            if data.empty:
                raise ValueError("Gaze data is empty.")
            return Stream(data)

        if homographies is None:
            if tag_info is None:
                raise ValueError(
                    "Marker information is required for homography estimation."
                )
            homographies = self.find_homographies(tag_info=tag_info)

        if synced_gaze is None:
            # Detect frequency / alignment
            if homographies.data.index.equals(self.gaze.data.index):
                synced_gaze = self.gaze
            else:
                synced_gaze = self.sync_gaze_to_video()

        data = gaze_on_surface(synced_gaze.data, homographies.data)

        # Save gaze on surface data to CSV
        data.to_csv(gaze_on_surface_path, index=True)

        return Stream(data)

    def fixations_on_surface(
        self,
        gaze_on_surface: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Events:
        """
        Project fixation events into surface space by summarizing gaze samples.

        This function maps each fixation to surface coordinates by averaging the
        surface-transformed gaze points (`x_trans`, `y_trans`) associated with
        that fixation. If saved data exists and `overwrite` is False, it is loaded
        from disk instead of being recomputed.

        Parameters
        ----------
        gaze_on_surface : pandas.DataFrame, optional
            DataFrame of gaze coordinates already transformed to surface space.
            If None, will be computed via `self.gaze_on_surface()`.
            Must include 'fixation id', 'x_trans', and 'y_trans' columns.
        overwrite : bool, optional
            If True, forces recomputation and overwrites any existing output file.
            Default is False.
        output_path : str or pathlib.Path, optional
            Custom path to save the resulting fixation data as a CSV.
            If None, defaults to `self.der_dir / "fixations_on_surface.csv"`.

        Returns
        -------
        Events
            An events object containing:
                - All columns from the raw fixations table
                - 'gaze x [surface coord]' : float
                    Mean surface-space x-coordinate for the fixation.
                - 'gaze y [surface coord]' : float
                    Mean surface-space y-coordinate for the fixation.
        """

        if output_path is None:
            fixation_on_surface_path = self.der_dir / "fixations_on_surface.csv"
        else:
            fixation_on_surface_path = Path(output_path)

        # Check if fixations already exist
        if fixation_on_surface_path.is_file() and not overwrite:
            data = pd.read_csv(fixation_on_surface_path)
            if data.empty:
                raise ValueError("Fixations data is empty.")
            return Events(data)

        raw_fixations = self.fixations.data

        if raw_fixations.empty:
            raise ValueError("No fixations data found.")

        if gaze_on_surface is None:
            # Check if gaze on surface already exists
            gaze_on_surface = self.gaze_on_surface()

        # Summarize gaze points first:
        gaze_means = (
            gaze_on_surface.data.groupby("fixation id", as_index=False)[
                ["gaze x [surface coord]", "gaze y [surface coord]"]
            ]
            .mean()
            .rename(
                columns={
                    "gaze x [surface coord]": "fixation x [surface coord]",
                    "gaze y [surface coord]": "fixation y [surface coord]",
                }
            )
        )

        raw_fixations = raw_fixations.reset_index(drop=False)

        # Merge back into fixations:
        data = raw_fixations.merge(gaze_means, on="fixation id", how="outer")
        data = data.set_index("start timestamp [ns]")

        # save fixations to csv
        data.to_csv(fixation_on_surface_path, index=True)

        return Events(data)

    def _estimate_camera_pose(
        self,
        tag_locations_df: pd.DataFrame,
        all_detections: Optional[Stream] = None,
        output_path: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Stream:
        """
        Compute the camera pose (R|t) for every frame.

        Parameters
        ----------
        tag_locations_df : pandas.DataFrame
            3-D positions, normals and size for every marker (columns:
            'tag_id','x','y','z','normal_x','normal_y','normal_z','size').
        all_detections : Stream, optional
            Per-frame marker detections.  If ``None``, they are produced by Recording.detect_markers().
        output_path : str or pathlib.Path, optional
            Path to save the resulting camera pose data as a JSON file. Defaults to `<der_dir>/camera_pose.json`.
        overwrite : bool, optional
            If True, forces recomputation and overwrites any existing saved result. Default is False.

        Returns
        -------
        Stream
            Indexed by "timestamp [ns]" with columns
            ``'frame id', 'translation_vector', 'rotation_vector', 'camera_pos'``.
        """

        json_file = (
            Path(output_path) if output_path else self.der_dir / "camera_pose.json"
        )

        # ------------------------------------------------------------------ load
        if json_file.is_file() and not overwrite:
            print(f"Loading saved camera pose from {json_file}")
            df = pd.read_json(json_file, orient="records", lines=True)
            df["timestamp [ns]"] = df["timestamp [ns]"].astype("int64")
            df = df.set_index("timestamp [ns]")
            if df.empty:
                raise ValueError("Camera pose data is empty.")
            return Stream(df)

        # ------------------------------------------------------------------ prerequisites
        req = {"tag_id", "pos_vec", "norm_vec", "size"}
        missing = req - set(tag_locations_df.columns)
        if missing:
            raise ValueError(f"tag_locations_df is missing: {missing}")

        if all_detections is None:
            all_detections = self.detect_markers()

        if all_detections.data.empty:
            raise ValueError("No marker detections found.")

        # ------------------------------------------------------------------ compute
        cam_pose_df = estimate_camera_pose(
            video=self.scene_video,
            tag_locations_df=tag_locations_df,
            all_detections=all_detections.data,
        )
        cam_pose_df.index.name = "timestamp [ns]"

        # ------------------------------------------------------------------ save
        cam_pose_df.reset_index().to_json(json_file, orient="records", lines=True)

        return Stream(cam_pose_df)

    def _smooth_camera_pose(
        self,
        camera_pose_raw: Optional[pd.DataFrame] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Kalman-smooth camera positions and gate outliers.

        Parameters
        ----------
        camera_pose_raw : pandas.DataFrame, optional
            Raw camera-pose table with columns ``'frame id'`` and ``'camera_pos'``.
            If *None*, tries to load *camera_pose.json* from ``self.der_dir`` or
            computes it via Recording.estimate_camera_pose().
        overwrite : bool, default False
            Recompute even if a smoothed file already exists.
        output_path : str or pathlib.Path, optional
            Where to save the JSON (``smoothed_camera_pose.json`` by default).
        kwargs : dict, optional
            Optional arguments:
                - initial_state_noise: float (default=0.1)
                - process_noise: float (default=0.1)
                - measurement_noise: float (default=0.01)
                - gating_threshold: float (default=2.0)
                - bidirectional: bool (default=False)

        Returns
        -------
        pd.DataFrame
            Same rows as *camera_pose_raw* with the extra column
            ``'smoothed_camera_pos'`` (3-vector).
        """

        # Defaults
        initial_state_noise = kwargs.get("initial_state_noise", 0.1)
        process_noise = kwargs.get("process_noise", 0.1)
        measurement_noise = kwargs.get("measurement_noise", 0.01)
        gating_threshold = kwargs.get("gating_threshold", 2.0)
        bidirectional = kwargs.get("bidirectional", False)

        # ------------------------------------------------ target path
        json_file = (
            Path(output_path)
            if output_path
            else self.der_dir / "smoothed_camera_pose.json"
        )

        # ------------------------------------------------ fast-path load
        if json_file.is_file() and not overwrite:
            print(f"Loading smoothed camera pose from {json_file}")
            df = pd.read_json(json_file, orient="records", lines=True)
            df["camera_pos"] = df["camera_pos"].apply(np.array)
            df["smoothed_camera_pos"] = df["smoothed_camera_pos"].apply(np.array)
            return df

        # ------------------------------------------------ obtain raw pose
        if camera_pose_raw is None:
            raw_path = self.der_dir / "camera_pose.json"
            if raw_path.is_file():
                camera_pose_raw = pd.read_json(raw_path, orient="records", lines=True)
                camera_pose_raw["camera_pos"] = camera_pose_raw["camera_pos"].apply(
                    lambda p: np.array(p, dtype=float)
                )
            else:
                camera_pose_raw = self.estimate_camera_pose()  # returns DataFrame

        if camera_pose_raw.empty:
            raise ValueError("Camera-pose table is empty; cannot smooth.")

        # ------------------------------------------------ compute
        smoothed = smooth_camera_pose(
            camera_pose_raw,
            initial_state_noise,
            process_noise,
            measurement_noise,
            gating_threshold,
            bidirectional,
        )

        # ------------------------------------------------ save & return
        smoothed.to_json(json_file, orient="records", lines=True)
        return smoothed

    def overlay_scanpath(
        self,
        scanpath: Optional[Stream] = None,
        show_video: bool = False,
        video_output_path: Optional[str | Path] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Render a video with the scan-path overlaid.

        Parameters
        ----------
        scanpath : Stream, optional
            Nested scan-path table (as from Recording.estimate_scanpath()).
            If *None*, it is loaded or computed automatically.
        show_video : bool
            Display the video live while rendering.
        video_output_path : str or pathlib.Path, optional
            Target MP4 path. Defaults to `<der_dir>/scanpath.mp4`.
            If *None*, no file is written.
        overwrite : bool, default False
            Regenerate the overlay even if the MP4 already exists.
        kwargs : dict, optional
            Optional arguments:
                - circle_radius: int, default 10
                    Radius of the fixation circles.
                - line_thickness: int, default 2
                    Thickness of the fixation lines.
                - text_size: int, default 1
                    Size of the fixation text.
                - max_fixations: int, default 10
                    Maximum number of fixations to display.
        """

        # Defaults for kwargs
        circle_radius = kwargs.get("circle_radius", 10)
        line_thickness = kwargs.get("line_thickness", 2)
        text_size = kwargs.get("text_size", 1)
        max_fixations = kwargs.get("max_fixations", 10)

        if video_output_path is None:
            video_output_path = self.der_dir / "scanpath.mp4"
        else:
            video_output_path = Path(video_output_path)

        if scanpath is None:
            scanpath = self.estimate_scanpath()

        if video_output_path.is_file() and not overwrite:
            print(
                f"Overlay video already exists at {video_output_path}; skipping render."
            )
            if show_video:
                print("`show_video=True` has no effect because rendering was skipped.")
            return

        if self.scene_video is None:
            raise ValueError("A loaded video is required to draw the overlay.")

        overlay_scanpath(
            self.scene_video,
            scanpath.data,
            circle_radius,
            line_thickness,
            text_size,
            max_fixations,
            show_video,
            video_output_path,
        )

    def overlay_detections_and_pose(
        self,
        april_detections: pd.DataFrame,
        camera_positions: pd.DataFrame,
        room_corners: np.ndarray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
        video_output_path: Path | str | None = None,
        graph_size: np.ndarray = np.array([300, 300]),
        show_video: bool = True,
    ):
        """
        Overlay AprilTag detections and camera poses on the video frames. The resulting video can be displayed and/or saved.

        Parameters
        ----------
        april_detections : pandas.DataFrame
            DataFrame containing the AprilTag detections.
        camera_positions : pandas.DataFrame
            DataFrame containing the camera positions.
        room_corners : numpy.ndarray
            Array containing the room corners coordinates. Defaults to a unit square.
        video_output_path pathlib.Path or str
            Path to save the video with detections and poses overlaid. Defaults to 'detection_and_pose.mp4'.
        graph_size : numpy.ndarray
            Size of the graph to overlay on the video. Defaults to [300, 300].
        show_video : bool
            Whether to display the video with detections and poses overlaid. Defaults to True.
        """
        if self.scene_video is None:
            raise ValueError(
                "Overlaying detections and pose on video requires video data."
            )

        if video_output_path is None:
            video_output_path = self.der_dir / "detection_and_pose.mp4"

        overlay_detections_and_pose(
            self,
            april_detections,
            camera_positions,
            room_corners,
            video_output_path,
            graph_size,
            show_video,
        )

    def clear_der_dir(
        self,
        include: str | list[str] = ["all"],
        exclude: str | list[str] = [],
    ):
        """
        Clear the derived data directory by removing files and folders.

        Parameters
        ----------
        include : str or list of str, optional
            Files or folders to delete. If ["all"], delete everything in the directory.
            Supports full names or base names without extensions (e.g. "scanpath" matches "scanpath.pkl").
        exclude : str or list of str, optional
            Files or folders to exclude. Applies only if include is ["all"].
            Also supports base names.
        """

        der_dir = Path(self.der_dir)
        if not der_dir.is_dir():
            raise ValueError(f"Derived data directory {der_dir} does not exist.")

        include_set = {include} if isinstance(include, str) else set(include)
        exclude_set = {exclude} if isinstance(exclude, str) else set(exclude)

        def matches_name(p: Path, names: set[str]) -> bool:
            return p.name in names or p.stem in names

        if "all" in include:
            # Delete everything not matching excluded names
            targets = [p for p in der_dir.iterdir() if not matches_name(p, exclude_set)]
        else:
            # Delete only explicitly included files/folders
            targets = [p for p in der_dir.iterdir() if matches_name(p, include_set)]

        deleted_paths = []
        for p in targets:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            deleted_paths.append(p.name)

        print(f"Deleted {len(deleted_paths)} items from {der_dir}: {deleted_paths}")

    def export_cloud_format(
        self,
        target_dir: str | Path,
        rebase: bool = True,
    ):
        """
        Export native data to cloud-like format.

        Parameters
        ----------
        target_dir : str or pathlib.Path
            Output directory to save the Cloud-Format structured data.
        rebase : bool, optional
            If True, re-initialize the recording on the target directory after export.
        """
        export_cloud_format(self, target_dir)
        if rebase:
            self.__init__(target_dir)

    def export_motion_bids(
        self,
        motion_dir: str | Path,
        prefix: Optional[str] = None,
        extra_metadata: dict = {},
    ):
        """
        Export IMU data to Motion-BIDS format. Continuous samples are saved to a .tsv
        file and metadata (with template fields) are saved to a .json file.
        Users should later edit the metadata file according to the experiment to make
        it BIDS-compliant.

        Parameters
        ----------
        motion_dir : str or pathlib.Path
            Output directory to save the Motion-BIDS formatted data.
        prefix : str, optional
            Prefix for the BIDS filenames, by default "sub-``wearer_name``_task-XXX_tracksys-NeonIMU".
            The format should be "sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]"
            (Fields in [] are optional). Files will be saved as
            ``{prefix}_motion.<tsv|json>``.
        extra_metadata : dict, optional
            Extra metadata to include in the .json file. Keys must be valid BIDS fields.

        Notes
        -----
        Motion-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
        standardize the organization of motion data for reproducible research [1]_.
        For more information, see
        https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html.

        References
        ----------
        .. [1] Jeung, S., Cockx, H., Appelhoff, S., Berg, T., Gramann, K., Grothkopp, S., ... & Welzel, J. (2024). Motion-BIDS: an extension to the brain imaging data structure to organize motion data for reproducible research. *Scientific Data*, 11(1), 716.
        """
        export_motion_bids(self, motion_dir, prefix, extra_metadata)

    def export_eye_bids(
        self,
        output_dir: str | Path,
        prefix: str = "",
        extra_metadata: dict = {},
    ):
        """
        Export eye-tracking data to Eye-tracking-BIDS format. Continuous samples
        and events are saved to .tsv.gz files with accompanying .json metadata files.
        Users should later edit the metadata files according to the experiment.

        Parameters
        ----------

        output_dir : str or pathlib.Path
            Output directory to save the Eye-tracking-BIDS formatted data.
        prefix : str, optional
            Prefix for the BIDS filenames, by default "sub-XX_recording-eye".
            The format should be `<matches>[_recording-<label>]_<physio|physioevents>.<tsv.gz|json>`
            (Fields in [] are optional). Files will be saved as
            ``{prefix}_physio.<tsv.gz|json>`` and ``{prefix}_physioevents.<tsv.gz|json>``.

        Notes
        -----
        Eye-tracking-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
        standardize the organization of eye-tracking data for reproducible research.
        The extension is still being finalized. This method follows the latest standards
        outlined in https://github.com/bids-standard/bids-specification/pull/1128.
        """
        export_eye_bids(self, output_dir, prefix, extra_metadata)
