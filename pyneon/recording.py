from pathlib import Path
from typing import Literal, Optional, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from numbers import Number
from functools import cached_property
import shutil

from .stream import Stream
from .events import Events
from .preprocess import concat_streams, concat_events, smooth_camera_pose
from .video import (
    SceneVideo,
    estimate_scanpath,
    detect_apriltags,
    estimate_camera_pose,
    find_homographies,
    transform_gaze_to_screen,
)
from .vis import plot_distribution, overlay_scanpath, overlay_detections_and_pose
from .export import export_motion_bids, export_eye_bids
from .utils import load_or_compute


def _check_file(dir_path: Path, stem: str):
    csv = dir_path / f"{stem}.csv"
    if csv.is_file():
        return True, csv.name, csv
    elif len(files := sorted(dir_path.glob(f"{stem}*"))) > 0:
        files_name = ", ".join([f.name for f in files])
        return True, files_name, files
    else:
        return False, None, None


class Recording:
    """
    Data from a single recording. The recording directory could be downloaded from
    either a single recording or a project on Pupil Cloud. In either case, the directory
    must contain an ``info.json`` file. For example, a recording directory could have the
    following structure:

    .. code-block:: text

        recording_dir/
        ├── info.json (REQUIRED)
        ├── gaze.csv
        ├── 3d_eye_states.csv
        ├── imu.csv
        ├── blinks.csv
        ├── fixations.csv
        ├── saccades.csv
        ├── events.csv
        ├── labels.csv
        ├── world_timestamps.csv
        ├── scene_camera.json
        ├── <scene_video>.mp4
        └── derivatives/ (PyNeon-generated derivatives)
            └── ...

    Streams, events, (and scene video) will be located but not loaded until
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
    info : dict
        Information about the recording. Read from ``info.json``. For details, see
        https://docs.pupil-labs.com/neon/data-collection/data-format/#info-json.
    start_time : int
        Start time (in ns) of the recording as in ``info.json``.
        May not match the start time of each data stream.
    start_datetime : datetime.datetime
        Start time (datetime) of the recording as in ``info.json``.
        May not match the start time of each data stream.
    contents : pandas.DataFrame
        Contents of the recording directory. Each index is a stream or event name
        (e.g. ``gaze`` or ``imu``) and columns are ``exist`` (bool),
        ``filename`` (str), and ``path`` (Path).
    """

    def __init__(self, recording_dir: str | Path):
        recording_dir = Path(recording_dir)
        if not recording_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {recording_dir}")
        if not (info_path := recording_dir / "info.json").is_file():
            raise FileNotFoundError(f"info.json not found in {recording_dir}")

        with open(info_path) as f:
            self.info = json.load(f)
        self.start_time = int(self.info["start_time"])
        self.start_datetime = datetime.fromtimestamp(self.start_time / 1e9)

        self.recording_id = self.info["recording_id"]
        self.recording_dir = recording_dir

        self.der_dir = recording_dir / "derivatives"
        if not self.der_dir.is_dir():
            self.der_dir.mkdir()

        self._get_contents()

    def __repr__(self) -> str:
        return f"""
Recording ID: {self.recording_id}
Wearer ID: {self.info["wearer_id"]}
Wearer name: {self.info["wearer_name"]}
Recording start time: {self.start_datetime}
Recording duration: {self.info["duration"] / 1e9}s
{self.contents.to_string()}
"""

    def _get_contents(self):
        contents = pd.DataFrame(
            index=[
                "3d_eye_states",
                "blinks",
                "events",
                "fixations",
                "gaze",
                "imu",
                "labels",
                "saccades",
                "world_timestamps",
                "scene_video_info",
                "scene_video",
            ],
            columns=["exist", "filename", "path"],
        )
        # Check for CSV files
        for stem in contents.index:
            contents.loc[stem, :] = _check_file(self.recording_dir, stem)

        # Check for scene video
        if len(video_path := list(self.recording_dir.glob("*.mp4"))) == 1:
            contents.loc["scene_video", :] = (True, video_path[0].name, video_path[0])
            if (camera_info := self.recording_dir / "scene_camera.json").is_file():
                contents.loc["scene_video_info", :] = (
                    True,
                    camera_info.name,
                    camera_info,
                )
            else:
                raise FileNotFoundError(
                    "Scene video has no accompanying scene_camera.json in "
                    f"{self.recording_dir}"
                )
        elif len(video_path) > 1:
            raise FileNotFoundError(
                f"Multiple scene video files found in {self.recording_dir}"
            )
        self.contents = contents

    @cached_property
    def gaze(self) -> Optional[Stream]:
        """
        Returns a (cached) :class:`pyneon.Stream` object containing gaze data or
        ``None`` if no ``gaze.csv`` is present.
        """
        if self.contents.loc["gaze", "exist"]:
            return Stream(self.contents.loc["gaze", "path"], 200)
        else:
            warnings.warn("Gaze data not loaded because no file was found.")

    @cached_property
    def imu(self) -> Optional[Stream]:
        """
        Returns a (cached) :class:`pyneon.Stream` object containing IMU data
        or ``None`` if no ``imu.csv`` is present.
        """
        if self.contents.loc["imu", "exist"]:
            return Stream(self.contents.loc["imu", "path"], 110)
        else:
            warnings.warn("IMU data not loaded because no file was found.")
        return None

    @cached_property
    def eye_states(self) -> Optional[Stream]:
        """
        Returns a (cached) :class:`pyneon.Stream` object containing eye states data
        or ``None`` if no ``3d_eye_states.csv`` is present.
        """
        if self.contents.loc["3d_eye_states", "exist"]:
            return Stream(self.contents.loc["3d_eye_states", "path"], 200)
        else:
            warnings.warn("3D eye states data not loaded because no file was found.")
            return None

    @cached_property
    def blinks(self) -> Optional[Events]:
        """
        Returns a (cached) :class:`pyneon.Events` object containing blinks data
        or ``None`` if no ``blinks.csv`` is present.
        """
        if self.contents.loc["blinks", "exist"]:
            return Events(self.contents.loc["blinks", "path"], "blinks", "blink id")
        else:
            warnings.warn("Blinks data not loaded because no file was found.")
            return None

    @cached_property
    def fixations(self) -> Optional[Events]:
        """
        Returns a (cached) :class:`pyneon.Events` object containing fixations data
        or ``None`` if no ``fixations.csv`` is present.
        """
        if self.contents.loc["fixations", "exist"]:
            return Events(
                self.contents.loc["fixations", "path"],
                "fixations",
                "fixation id",
            )
        else:
            warnings.warn("Fixations data not loaded because no file was found.")
            return None

    @cached_property
    def saccades(self) -> Optional[Events]:
        """
        Returns a (cached) :class:`pyneon.Events` object containing saccades data
        or ``None`` if no ``saccades.csv`` is present.
        """
        if self.contents.loc["saccades", "exist"]:
            return Events(
                self.contents.loc["saccades", "path"], "saccades", "saccade id"
            )
        else:
            warnings.warn("Saccades data not loaded because no file was found.")
            return None

    @cached_property
    def events(self) -> Optional[Events]:
        """
        Returns a (cached) :class:`pyneon.Events` object containing events data
        or ``None`` if no ``events.csv`` is present.
        """
        if self.contents.loc["events", "exist"]:
            events_file = self.contents.loc["events", "path"]
            return Events(events_file, "events")
        else:
            warnings.warn("Events data not loaded because no file was found.")
            return None

    @cached_property
    def video(self) -> Optional[SceneVideo]:
        """
        Returns a (cached) :class:`pyneon.SceneVideo` object containing scene video data
        or ``None`` if no ``scene_video.mp4`` is present.
        """
        if (
            (video_file := self.contents.loc["scene_video", "path"])
            and (timestamp_file := self.contents.loc["world_timestamps", "path"])
            and (video_info_file := self.contents.loc["scene_video_info", "path"])
        ):
            return SceneVideo(video_file, timestamp_file, video_info_file)
        else:
            warnings.warn(
                "Scene video not loaded because not all video-related files "
                "(video, scene_camera.json, world_timestamps.csv) are found."
            )
            return None

    def concat_streams(
        self,
        stream_names: str | list[str],
        sampling_freq: Number | str = "min",
        interp_float_kind: str = "linear",
        interp_other_kind: str = "nearest",
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
            (``"3d_eye_states"``) is also tolerated as an alias for ``"eye_states"``).
        sampling_freq : float or int or str, optional
            Sampling frequency of the concatenated streams.
            If numeric, the streams will be interpolated to this frequency.
            If ``"min"`` (default), the lowest nominal sampling frequency
            of the selected streams will be used.
            If ``"max"``, the highest nominal sampling frequency will be used.
        interp_float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
        interp_other_kind : str, optional
            Kind of interpolation applied on columns of other types.
            Defaults to ``"nearest"``.
        inplace : bool, optional
            Replace selected stream data with interpolated data during concatenation
            if ``True``. Defaults to ``False``.

        Returns
        -------
        pandas.DataFrame
            Concatenated data.
        """
        new_data = concat_streams(
            self,
            stream_names,
            sampling_freq,
            interp_float_kind,
            interp_other_kind,
            inplace,
        )
        return Stream(new_data)

    def concat_events(self, event_names: str | list[str]) -> pd.DataFrame:
        """
        Concatenate different events. All columns in the selected event type will be
        present in the final DataFrame. An additional ``"type"`` column denotes the event
        type. If ``events`` is selected, its ``"timestamp [ns]"`` column will be
        renamed to ``"start timestamp [ns]"``, and the ``"name`` and ``"type"`` columns will
        be renamed to ``"message name"`` and ``"message type"`` respectively to provide
        a more readable output.

        Parameters
        ----------
        event_names : list of str
            List of event names to concatenate. Event names must be in
            ``{"blinks", "fixations", "saccades", "events"}``
            (singular forms are tolerated).

        Returns
        -------
        pandas.DataFrame
            Concatenated events.
        """
        return concat_events(self, event_names)

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
                - 'frame_idx': Index of the video frame corresponding to the gaze data
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

        if self.gaze is None or self.video is None:
            raise ValueError("Gaze-video synchronization requires gaze and video data.")

        synced_gaze = self.gaze.window_average(self.video.ts, window_size).data
        synced_gaze["frame_idx"] = np.arange(len(synced_gaze))

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
            recording.sync_gaze_to_video().
        lk_params : dict, optional
            Parameters forwarded to the LK optical-flow call.
        output_path : str or pathlib.Path, optional
            Where to save the pickle. Defaults to ``<der_dir>/scanpath.pkl``.
        overwrite : bool, optional
            Force recomputation even if a pickle exists.

        Returns
        -------
        Stream
            Indexed by ``"timestamp [ns]"`` with one column ``"fixations"``
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
            return Stream(df, sampling_freq_nominal=int(self.video.fps))

        if sync_gaze is None:
            sync_gaze = self.sync_gaze_to_video()

        scanpath_df = estimate_scanpath(
            self.video,
            sync_gaze,
            lk_params=lk_params,
        )
        scanpath_df.index.name = "timestamp [ns]"

        # ------------------------------------------------------------------ save
        scanpath_df.to_pickle(scanpath_path)

        return Stream(scanpath_df, sampling_freq_nominal=int(self.video.fps))

    def detect_apriltags(
        self,
        tag_family: str = "tag36h11",
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
        **kwargs,
    ) -> Stream:
        """
        Detect AprilTags in a video and return their positions per frame.

        Runs AprilTag detection on video frames using the `pupil_apriltags` backend.
        Uses saved results if available unless `overwrite=True`.

        Parameters
        ----------
        tag_family : str, optional
            The AprilTag family to detect (e.g., "tag36h11"). Default is "tag36h11".
        overwrite : bool, optional
            If True, reruns detection even if saved results exist.
        output_path : str or pathlib.Path, optional
            Path to save the detection JSON file. Defaults to `<der_dir>/apriltags.json`.
        **kwargs : keyword arguments
            Additional parameters for AprilTag detection, including:
                - 'nthreads': Number of threads to use for detection. Default is 4.
                - 'quad_decimate': Decimation factor for the quad detection. Default is 1.0, thus no decimation.
                - 'skip_frames': Number of frames to skip between detections. Default is 1, thus no skipping.

        Returns
        -------
        Stream
            Stream indexed by `"timestamp [ns]"` with one row per detected tag, including:
                - 'frame_idx': Index in the downsampled video sequence
                - 'orig_frame_idx': Original frame index in the full video
                - 'tag_id': ID of the detected AprilTag
                - 'corners': A 4x2 array of tag corner coordinates
                - 'center': A 1x2 array of the tag center
        """

        nthreads = kwargs.get("nthreads", 4)
        quad_decimate = kwargs.get("quad_decimate", 1.0)
        skip_frames = kwargs.get("skip_frames", 1)

        if output_path is None:
            json_file = self.der_dir / "apriltags.json"
        else:
            json_file = Path(output_path)

        # If a saved file exists and overwrite is False, just read and return it
        if json_file.is_file() and not overwrite:
            print(f"Loading saved detections from {json_file}")
            all_detections = pd.read_json(json_file, orient="records", lines=True)
            all_detections["timestamp [ns]"] = all_detections["timestamp [ns]"].astype(
                "int64"
            )
            if all_detections.empty:
                raise ValueError("AprilTag detection data is empty.")
            return Stream(all_detections)

        all_detections = detect_apriltags(
            video=self.video,
            tag_family=tag_family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            skip_frames=skip_frames,
        )

        # Save results to JSON
        all_detections.reset_index().to_json(json_file, orient="records", lines=True)

        return Stream(all_detections)

    def find_homographies(
        self,
        marker_info: pd.DataFrame,
        all_detections: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
        **kwargs,
    ) -> Stream:
        """
        Compute and return homographies for each frame using AprilTag detections and reference marker layout.

        Parameters
        ----------
        marker_info : pandas.DataFrame
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
        **kwargs : keyword arguments
            Additional parameters for homography computation, including:
                - 'coordinate_system': Coordinate system for the homography ('opencv' or 'psychopy'). Default is 'opencv'.
                - 'screen_size': Size of the screen in pixels (width, height). Default is (1920, 1080).
                - 'skip_frames': Number of frames to skip between detections. Default is 1.
                - 'settings': Additional settings for the homography computation.

        Returns
        -------
        Stream
            A Stream object indexed by `"timestamp [ns]"` containing:
                - 'frame_idx': Video frame index
                - 'homography': 3x3 NumPy array representing the homography matrix for that frame
        """

        # Defaults for kwargs
        coordinate_system = kwargs.get("coordinate_system", "opencv")
        screen_size = kwargs.get("screen_size", (1920, 1080))
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
            homographies = Stream(df, sampling_freq_nominal=30)
            return homographies

        if all_detections is None:
            all_detections = self.detect_apriltags()

        if all_detections.data.empty:
            raise ValueError("No AprilTag detections found.")

        homographies_df = find_homographies(
            self.video,
            all_detections.data,
            marker_info.copy(deep=True),
            screen_size,
            skip_frames=skip_frames,
            coordinate_system=coordinate_system,
            settings=settings,
        )

        homographies_df.to_pickle(pkl_file)

        return Stream(homographies_df)

    def gaze_to_screen(
        self,
        homographies: Optional[Stream] = None,
        marker_info: Optional[pd.DataFrame] = None,
        synced_gaze: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Project gaze coordinates from eye space to screen space using homographies.

        Computes or loads frame-wise homographies and applies them to the synchronized
        gaze data to transform it into screen coordinates. If a saved version exists
        and `overwrite` is False, the data is loaded from disk.

        Parameters
        ----------
        homographies : Stream, optional
            Stream containing precomputed homographies. If None, they are computed from `marker_info`.
        marker_info : pandas.DataFrame, optional
            AprilTag marker info used to compute homographies. Required if `homographies` is None.
        synced_gaze : Stream, optional
            Gaze data aligned to video frames. If None, will be computed using `sync_gaze_to_video()`.
        overwrite : bool, optional
            If True, recompute and overwrite any existing screen-transformed gaze data. Default is False.
        output_path : str or pathlib.Path, optional
            File path to save the resulting CSV. Defaults to `<der_dir>/gaze_on_screen.csv`.

        Returns
        -------
        Stream
            A Stream containing gaze data with screen coordinates, including:
                - 'frame_idx': Frame index
                - 'x_trans', 'y_trans': Gaze coordinates in screen pixel space
                - Any additional columns from the synchronized gaze input
        """

        if output_path is None:
            gaze_on_screen_path = self.der_dir / "gaze_on_screen.csv"
        else:
            gaze_on_screen_path = Path(output_path)

        if gaze_on_screen_path.is_file() and not overwrite:
            # Load saved gaze on screen data
            gaze_on_screen = pd.read_csv(gaze_on_screen_path)
            if gaze_on_screen.empty:
                raise ValueError("Gaze data is empty.")
            return Stream(gaze_on_screen)

        if homographies is None:
            if marker_info is None:
                raise ValueError(
                    "Marker information is required for homography estimation."
                )
            homographies = self.find_homographies(marker_info=marker_info)

        if synced_gaze is None:
            # Check if synced gaze already exists
            synced_gaze = self.sync_gaze_to_video()

        gaze_on_screen = transform_gaze_to_screen(synced_gaze.data, homographies.data)

        # Save gaze on screen data to CSV
        gaze_on_screen.to_csv(gaze_on_screen_path, index=True)

        return Stream(gaze_on_screen)

    def fixations_to_screen(
        self,
        gaze_on_screen: Optional[Stream] = None,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Events:
        """
        Project fixation events into screen space by summarizing gaze samples.

        This function maps each fixation to screen coordinates by averaging the
        screen-transformed gaze points (`x_trans`, `y_trans`) associated with
        that fixation. If saved data exists and `overwrite` is False, it is loaded
        from disk instead of being recomputed.

        Parameters
        ----------
        gaze_on_screen : pandas.DataFrame, optional
            DataFrame of gaze coordinates already transformed to screen space.
            If None, will be computed via `self.gaze_to_screen()`.
            Must include 'fixation id', 'x_trans', and 'y_trans' columns.
        overwrite : bool, optional
            If True, forces recomputation and overwrites any existing output file.
            Default is False.
        output_path : str or pathlib.Path, optional
            Custom path to save the resulting fixation data as a CSV.
            If None, defaults to `self.der_dir / "fixations_on_screen.csv"`.

        Returns
        -------
        Events
            An events object containing:
                - All columns from the raw fixations table
                - 'gaze x [screen px]' : float
                    Mean screen-space x-coordinate for the fixation.
                - 'gaze y [screen px]' : float
                    Mean screen-space y-coordinate for the fixation.
        """

        if output_path is None:
            fixation_on_screen_path = self.der_dir / "fixations_on_screen.csv"
        else:
            fixation_on_screen_path = Path(output_path)

        # Check if fixations already exist
        if fixation_on_screen_path.is_file() and not overwrite:
            fixation_on_screen = pd.read_csv(fixation_on_screen_path)
            if fixation_on_screen.empty:
                raise ValueError("Fixations data is empty.")
            return Events(fixation_on_screen, id_name="fixation id")

        raw_fixations = self.fixations.data

        if raw_fixations.empty:
            raise ValueError("No fixations data found.")

        if gaze_on_screen is None:
            # Check if gaze on screen already exists
            gaze_on_screen = self.gaze_to_screen()

        # Summarize gaze points first:
        gaze_means = (
            gaze_on_screen.data.groupby("fixation id", as_index=False)[
                ["x_trans", "y_trans"]
            ]
            .mean()
            .rename(
                columns={
                    "x_trans": "gaze x [screen px]",
                    "y_trans": "gaze y [screen px]",
                }
            )
        )

        raw_fixations = raw_fixations.reset_index(drop=False)

        # Merge back into fixations:
        fixation_on_screen = raw_fixations.merge(
            gaze_means, on="fixation id", how="outer"
        )
        fixation_on_screen = fixation_on_screen.set_index("start timestamp [ns]")

        # save fixations to csv
        fixation_on_screen.to_csv(fixation_on_screen_path, index=True)

        return Events(fixation_on_screen)

    def estimate_camera_pose(
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
            3-D positions, normals and size for every AprilTag (columns:
            'tag_id','x','y','z','normal_x','normal_y','normal_z','size').
        all_detections : Stream, optional
            Per-frame AprilTag detections.  If ``None``, they are produced by
            recording.detect_aptiltags().
        output_path : str or pathlib.Path, optional
            Path to save the resulting camera pose data as a JSON file. Defaults to `<der_dir>/camera_pose.json`.
        overwrite : bool, optional
            If True, forces recomputation and overwrites any existing saved result. Default is False.

        Returns
        -------
        Stream
            Indexed by ``"timestamp [ns]"`` with columns
            ``'frame_idx', 'translation_vector', 'rotation_vector', 'camera_pos'``.
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
            return Stream(df, sampling_freq_nominal=int(self.video.fps))

        # ------------------------------------------------------------------ prerequisites
        req = {"tag_id", "pos_vec", "norm_vec", "size"}
        missing = req - set(tag_locations_df.columns)
        if missing:
            raise ValueError(f"tag_locations_df is missing: {missing}")

        if all_detections is None:
            all_detections = self.detect_apriltags()

        if all_detections.data.empty:
            raise ValueError("No AprilTag detections found.")

        # ------------------------------------------------------------------ compute
        cam_pose_df = estimate_camera_pose(
            video=self.video,
            tag_locations_df=tag_locations_df,
            all_detections=all_detections.data,
        )
        cam_pose_df.index.name = "timestamp [ns]"

        # ------------------------------------------------------------------ save
        cam_pose_df.reset_index().to_json(json_file, orient="records", lines=True)

        return Stream(cam_pose_df, sampling_freq_nominal=int(self.video.fps))

    def smooth_camera_pose(
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
            Raw camera-pose table with columns ``'frame_idx'`` and ``'camera_pos'``.
            If *None*, tries to load *camera_pose.json* from ``self.der_dir`` or
            computes it via recording.estimate_camera_pose.
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
            Nested scan-path table (as from recording.estimate_scanpath).
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

        if self.video is None:
            raise ValueError("A loaded video is required to draw the overlay.")

        overlay_scanpath(
            self.video,
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
        if self.video is None:
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
            Both full names and base names (without extension) are accepted.
        exclude : str or list of str, optional
            Files or folders to exclude from deletion. Only applies if include == ["all"].
        """
        der_dir = Path(self.der_dir)
        if not der_dir.is_dir():
            raise ValueError(f"Derived data directory {der_dir} does not exist.")

        # Normalize inputs to sets of strings
        to_set = lambda x: {x} if isinstance(x, str) else set(x)
        include, exclude = to_set(include), to_set(exclude)

        def name_matches(p: Path, names: set[str]) -> bool:
            return p.name in names or p.stem in names

        items = list(der_dir.iterdir())
        targets = [
            p for p in items
            if ("all" in include and not name_matches(p, exclude)) or
            ("all" not in include and name_matches(p, include))
        ]

        for p in targets:
            (shutil.rmtree if p.is_dir() else p.unlink)()

        print(f"Deleted {len(targets)} items from {der_dir}: {[p.name for p in targets]}")


    def export_motion_bids(
        self,
        motion_dir: str | Path,
        prefix: str = "",
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
            Prefix for the BIDS filenames, by default "sub-XX_task-YY_tracksys-NeonIMU".
            The format should be `sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]`
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
        The extension is still being finialized. This method follows the latest standards
        outlined in https://github.com/bids-standard/bids-specification/pull/1128.
        """
        export_eye_bids(self, output_dir, prefix, extra_metadata)
