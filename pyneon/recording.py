from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from numbers import Number
from functools import cached_property

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
        interp_float_kind: str = "cubic",
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
            Defaults to ``"cubic"``. For details see :class:`scipy.interpolate.interp1d`.
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
        step_size: int = 10,
        sigma: int | float = 2,
        width_height: tuple[int, int] = (1600, 1200),
        cmap: str = "inferno",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
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
        step_size : int
            Size of the grid cells in pixels. Defaults to 10.
        sigma : int or float
            Standard deviation of the Gaussian kernel used to smooth the heatmap.
            If None or 0, no smoothing is applied. Defaults to 2.
        width_height : tuple[int, int]
            If video is not available, the width and height of the scene camera frames to
            specify the heatmap dimensions. Defaults to (1600, 1200).
        cmap : str
            Colormap to use for the heatmap. Defaults to 'inferno'.
        ax : matplotlib.axes.Axes or None
            Axis to plot the frame on. If ``None``, a new figure is created.
            Defaults to ``None``.
        show : bool
            Show the figure if ``True``. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.
        ax : matplotlib.axes.Axes
            Axis object containing the plot.
        """
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
            If True, force recomputation even if cached data exists. Default is False.
        output_path : str or Path, optional
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

        ### TODO: optimize window perfomance
        synced_gaze = self.gaze.window_average(self.video.ts, window_size).data
        synced_gaze["frame_idx"] = np.arange(len(synced_gaze))

        synced_gaze.index.name = "timestamp [ns]"
        synced_gaze.to_csv(gaze_file, index=True)


        ### TODO: unknown columns in the output, will raise warnings ("frame_idx")
        return Stream(synced_gaze)

    def estimate_scanpath(
        self,
        sync_gaze: Optional[Stream] = None,
        lk_params: Optional[dict] = None,
        output_path: Optional[str | Path] = None,
        overwrite : bool = False,
    ) -> Stream:
        """
        Estimate scanpaths by mapping fixations across video frames using optical flow.

        Computes frame-by-frame gaze propagation using the Lucas-Kanade algorithm, based on
        synchronized gaze data. If a cached result exists and `overwrite` is False, it is loaded
        from disk. The result includes nested fixation information for each frame.

        Parameters
        ----------
        sync_gaze : Stream, optional
            Gaze data synchronized to video frames. If None, computed via `sync_gaze_to_video()`.
        lk_params : dict, optional
            Parameters passed to the Lucas-Kanade optical flow method. Defaults to recommended values.
        output_path : str or Path, optional
            Path to save the output file. Defaults to `<der_dir>/scanpath.pkl`.
        overwrite : bool, optional
            If True, recompute and overwrite any existing cached result.

        Returns
        -------
        Stream
            A Stream indexed by `"timestamp [ns]"`, containing:
                - 'fixations': A nested DataFrame of fixations per frame
        """

        if output_path is None:
            scanpath_path = self.der_dir / "scanpath.pkl"
        else:
            scanpath_path = Path(output_path)

        # Check if scanpath already exists
        if scanpath_path.is_file() and not overwrite:
            print(f"Loading cached scanpath from {scanpath_path}")
            scanpath = pd.read_pickle(scanpath_path)
            if scanpath.empty:
                raise ValueError("Scanpath data is empty.")
            return Stream(scanpath)

        if sync_gaze is None:
            sync_gaze = self.sync_gaze_to_video()

        scanpath = estimate_scanpath(
            self.video,
            sync_gaze,
            lk_params=lk_params,
        )

        # Save scanpath to pickle
        scanpath.to_pickle(scanpath_path)

        ### TODO: scanpath contains nested dataframes, which will yield warnings
        return Stream(scanpath)

    def detect_apriltags(
        self,
        tag_family: str = "tag36h11",
        nthreads: int = 4,
        quad_decimate: float = 1.0,
        skip_frames: int = 1,
        overwrite: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Detect AprilTags in a video and return their positions per frame.

        Runs AprilTag detection on video frames using the `pupil_apriltags` backend.
        Uses cached results if available unless `overwrite=True`.

        Parameters
        ----------
        tag_family : str, optional
            The AprilTag family to detect (e.g., "tag36h11"). Default is "tag36h11".
        nthreads : int, optional
            Number of threads to use for detection. Default is 4.
        quad_decimate : float, optional
            Downsampling factor applied before detection. Higher values increase speed and reduce accuracy. Default is 1.0.
        skip_frames : int, optional
            Process every N-th frame (1 = no skipping). Speeds up processing at the cost of temporal resolution.
        overwrite : bool, optional
            If True, reruns detection even if cached results exist.
        output_path : str or Path, optional
            Path to save the detection JSON file. Defaults to `<der_dir>/apriltags.json`.

        Returns
        -------
        Stream
            Stream indexed by `"timestamp [ns]"` with one row per detected tag, including:
                - 'frame_idx': Index in the downsampled video sequence
                - 'orig_frame_idx': Original frame index in the full video
                - 'tag_id': ID of the detected AprilTag
                - 'corners': A 4×2 array of tag corner coordinates
                - 'center': A 1×2 array of the tag center
        """
        if output_path is None:
            json_file = self.der_dir / "apriltags.json"
        else:
            json_file = Path(output_path)

        # If a cached file exists and overwrite is False, just read and return it
        if json_file.is_file() and not overwrite:
            print(f"Loading cached detections from {json_file}")
            all_detections = pd.read_json(json_file, orient="records", lines=True)
            all_detections["timestamp [ns]"] = all_detections["timestamp [ns]"].astype("int64")
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

        ### TODO: unknown columns in the output, will raise warnings ("processed_frame_idx", "frame_idx", "tag_id", "corners", "center")
        return Stream(all_detections)

    def find_homographies(
            self,
            marker_info: pd.DataFrame,
            all_detections: Optional[Stream] = None,
            coordinate_system: str = "opencv",
            screen_size: tuple[int, int] = (1920, 1080),
            settings: Optional[dict] = None,
            overwrite: bool = False,
            output_path: Optional[str | Path] = None,
    ) -> Stream:
        """
        Compute and return homographies for each frame using AprilTag detections and reference marker layout.

        Parameters
        ----------
        marker_info : pd.DataFrame
            DataFrame containing AprilTag reference positions and orientations, with columns:
                - 'tag_id': ID of the tag
                - 'x', 'y', 'z': 3D coordinates of the tag's center
                - 'normal_x', 'normal_y', 'normal_z': Normal vector of the tag surface
                - 'size': Side length of the tag in meters
        all_detections : Stream, optional
            Stream containing AprilTag detection results per frame. If None, detections are recomputed.
        coordinate_system : {"opencv", "psychopy"}, default="opencv"
            Defines the coordinate convention used for screen and marker layout:
                - "opencv": Origin top-left, y increases downward
                - "psychopy": Origin center, y increases upward
        screen_size : tuple of int, default=(1920, 1080)
            Pixel dimensions of the target screen.
        settings : dict, optional
            Optional parameters for homography computation (e.g., RANSAC thresholds).
        overwrite : bool, optional
            Whether to force recomputation even if cached homographies exist.
        output_path : str or Path, optional
            Optional file path for saving the homographies as JSON. If None, defaults to `<der_dir>/homographies.json`.

        Returns
        -------
        Stream
            A Stream object indexed by `"timestamp [ns]"` containing:
                - 'frame_idx': Video frame index
                - 'homography': 3x3 NumPy array representing the homography matrix for that frame
        """
        if output_path is None:
            pkl_file = self.der_dir / "homographies.pkl"
        else:
            pkl_file = Path(output_path)

        # If a cached file exists and overwrite is False, just read and return it
        if pkl_file.is_file() and not overwrite:
            print(f"Loading cached homographies from {pkl_file}")
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
            coordinate_system=coordinate_system,
            settings=settings,
        )

        homographies_df.to_pickle(pkl_file)
        ### TODO: unknown columns in the output, will raise warnings (frame_idx, homography)
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
        gaze data to transform it into screen coordinates. If a cached version exists
        and `overwrite` is False, the data is loaded from disk.

        Parameters
        ----------
        homographies : Stream, optional
            Stream containing precomputed homographies. If None, they are computed from `marker_info`.
        marker_info : pd.DataFrame, optional
            AprilTag marker info used to compute homographies. Required if `homographies` is None.
        synced_gaze : Stream, optional
            Gaze data aligned to video frames. If None, will be computed using `sync_gaze_to_video()`.
        overwrite : bool, optional
            If True, recompute and overwrite any existing screen-transformed gaze data. Default is False.
        output_path : str or Path, optional
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
            # Load cached gaze on screen data
            gaze_on_screen = pd.read_csv(gaze_on_screen_path)
            if gaze_on_screen.empty:
                raise ValueError("Gaze data is empty.")
            return Stream(gaze_on_screen)

        if homographies is None:
            if marker_info is None:
                raise ValueError("Marker information is required for homography estimation.")
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
        that fixation. If cached data exists and `overwrite` is False, it is loaded
        from disk instead of being recomputed.

        Parameters
        ----------
        gaze_on_screen : pd.DataFrame, optional
            DataFrame of gaze coordinates already transformed to screen space.
            If None, will be computed via `self.gaze_to_screen()`.
            Must include 'fixation id', 'x_trans', and 'y_trans' columns.
        overwrite : bool, optional
            If True, forces recomputation and overwrites any existing output file.
            Default is False.
        output_path : str or Path, optional
            Custom path to save the resulting fixation data as a CSV.
            If None, defaults to `self.der_dir / "fixations_on_screen.csv"`.

        Returns
        -------
        pd.DataFrame
            Fixation-level DataFrame indexed by 'start timestamp [ns]', containing:
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
        fixation_on_screen = raw_fixations.merge(gaze_means, on="fixation id", how="outer")
        fixation_on_screen = fixation_on_screen.set_index("start timestamp [ns]")

        # save fixations to csv
        fixation_on_screen.to_csv(fixation_on_screen_path, index=True)

        return Events(fixation_on_screen)

    def estimate_camera_pose(
        self,
        tag_locations_df: pd.DataFrame,
        all_detections: Optional[pd.DataFrame] = None,
        output_path: Optional[str | Path] = None, 
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the camera positions from AprilTag detections in a video.

        Parameters
        ----------
        tag_locations_df : pd.DataFrame
            A DataFrame containing AprilTag 3D locations, orientations, and sizes.
            Required columns:
                - 'tag_id': int, ID of the tag
                - 'x', 'y', 'z': float, coordinates of the tag's center
                - 'normal_x', 'normal_y', 'normal_z': float, components of the tag's normal vector
                - 'size': float, the side length of the tag in meters

        all_detections : pd.DataFrame, optional
            DataFrame containing AprilTag detections for each frame, with columns:
                - 'frame_idx': The frame number (int)
                - 'tag_id': The ID of the detected AprilTag (int)
                - 'corners': A (4x2) array of the tag corner pixel coordinates (np.ndarray)
                - 'center': A (1x2) array of the tag center pixel coordinates (np.ndarray)
            If empty, the detections are computed using the `detect_apriltags` function with default settings.

        output_path : str or Path, optional
            Path to save the resulting camera pose data as a JSON file. Defaults to `<der_dir>/camera_pose.json`.
        
        overwrite : bool, optional 
            If True, forces recomputation and overwrites any existing saved result. Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the camera positions for each frame, with columns:
            - 'frame_idx': The frame number
            - 'translation_vector': A (3,) array with the camera translation vector
            - 'rotation_vector': A (3,) array with the camera rotation vector (Rodrigues form)
            - 'camera_pos': A (3,) array with the camera position in world coordinates
        """

        required_columns = {
            "tag_id",
            "x",
            "y",
            "z",
            "normal_x",
            "normal_y",
            "normal_z",
            "size",
        }

        if not required_columns.issubset(tag_locations_df.columns):
            missing = required_columns - set(tag_locations_df.columns)
            raise ValueError(f"tag_locations_df is missing required columns: {missing}")


        if output_path is None:
            json_file = self.der_dir / "camera_pose.json"
        else:
            json_file = Path(output_path)

        # check for detections dataframe
        if all_detections == None:
            # Check if detections already exist
            all_detections = self.detect_apriltags()
            if all_detections.empty:
                raise ValueError("No AprilTag detections found.")
            

        # Check if result JSON already exists
        if json_file.is_file() and not overwrite:
            print(f"Loading cached camera pose from {json_file}")
            camera_pose = pd.read_json(json_file, orient="records", lines=True)
            if camera_pose.empty:
                raise ValueError("Camera pose data is empty.")
            return camera_pose

        # Compute camera positions
        camera_pose = estimate_camera_pose(
            video=self.video,
            tag_locations_df=tag_locations_df,
            all_detections=all_detections,
        )

        # Save to JSON
        camera_pose.to_json(json_file, orient="records")

        return camera_pose

    def smooth_camera_pose(
        self,
        camera_pose_raw: pd.DataFrame = pd.DataFrame(),
        initial_state_noise: float = 0.1,
        process_noise: float = 0.1,
        measurement_noise: float = 0.01,
        gating_threshold: float = 2.0,
        bidirectional: bool = False,
    ) -> pd.DataFrame:
        """
        Apply a Kalman filter to smooth camera positions and gate outliers based on Mahalanobis distance.
        Expects a DataFrame containing 'frame_idx' and 'camera_pos' columns, where 'camera_pos' is a
        length-3 array-like object representing [x, y, z] coordinates.

        Parameters
        ----------
        camera_pose_raw : pd.DataFrame
            DataFrame containing 'frame_idx' and 'camera_pos' columns.
        process_noise : float, optional
            Process noise covariance scaling factor. Default is 0.005.
        measurement_noise : float, optional
            Measurement noise covariance scaling factor. Default is 0.005.
        gating_threshold : float, optional
            Mahalanobis distance threshold for gating outliers. Default is 3.0.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the same 'frame_idx' as input and an additional column 'smoothed_camera_pos'
            containing the smoothed positions.
        """

        if camera_pose_raw.empty:
            # Check if JSON already exists
            if (json_file := self.der_dir / "camera_pose.json").is_file():
                camera_pose_raw = pd.read_json(json_file, orient="records", lines=True)
                # Ensure 'camera_pos' is parsed as NumPy arrays
                camera_pose_raw["camera_pos"] = camera_pose_raw["camera_pos"].apply(
                    lambda pos: np.array(pos, dtype=float)
                )
            else:
                # Run the function to get the data
                camera_pose_raw = self.estimate_camera_pose()

        smoothed_pose = smooth_camera_pose(
            camera_pose_raw,
            initial_state_noise,
            process_noise,
            measurement_noise,
            gating_threshold,
            bidirectional,
        )

        # Save to JSON
        smoothed_pose.to_json(
            self.der_dir / "smoothed_camera_pose.json", orient="records"
        )

        return smoothed_pose

    def overlay_scanpath(
        self,
        scanpath: Optional[pd.DataFrame] = None,
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
        if scanpath is None:
            scanpath = self.estimate_scanpath()
        if self.video is None:
            raise ValueError("Plotting scanpath on video requires video data.")

        overlay_scanpath(
            self,
            scanpath,
            circle_radius,
            line_thickness,
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
        april_detections : :class:`pandas.DataFrame`
            DataFrame containing the AprilTag detections.
        camera_positions : :class:`pandas.DataFrame`
            DataFrame containing the camera positions.
        room_corners : :class:`numpy.ndarray`
            Array containing the room corners coordinates. Defaults to a unit square.
        video_output_path : :class:`pathlib.Path` or str
            Path to save the video with detections and poses overlaid. Defaults to 'detection_and_pose.mp4'.
        graph_size : :class:`numpy.ndarray`
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
