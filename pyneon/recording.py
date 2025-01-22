from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from numbers import Number

from .stream import NeonGaze, NeonIMU, NeonEyeStates, CustomStream
from .events import NeonBlinks, NeonFixations, NeonSaccades, NeonEvents
from .preprocess import concat_streams, concat_events, smooth_camera_pose
from .video import NeonVideo, estimate_scanpath, detect_apriltags, estimate_camera_pose
from .vis import plot_distribution, overlay_scanpath, overlay_detections_and_pose
from .export import export_motion_bids, export_eye_bids


def _check_file(dir_path: Path, stem: str):
    csv = dir_path / f"{stem}.csv"
    if csv.is_file():
        return True, csv.name, csv
    elif len(files := sorted(dir_path.glob(f"{stem}*"))) > 0:
        files_name = ", ".join([f.name for f in files])
        return True, files_name, files
    else:
        return False, None, None


class NeonRecording:
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

        self._gaze = None
        self._eye_states = None
        self._imu = None
        self._blinks = None
        self._fixations = None
        self._saccades = None
        self._events = None
        self._video = None

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

    @property
    def gaze(self) -> Optional[NeonGaze]:
        """
        Returns a NeonGaze object or None if no gaze data is found.
        """
        if self._gaze is None:
            if self.contents.loc["gaze", "exist"]:
                gaze_file = self.contents.loc["gaze", "path"]
                self._gaze = NeonGaze(gaze_file)
            else:
                warnings.warn("Gaze data not loaded because no file was found.")
        return self._gaze

    @property
    def imu(self) -> Optional[NeonIMU]:
        """
        Returns a NeonIMU object or None if no IMU data is found.
        """
        if self._imu is None:
            if self.contents.loc["imu", "exist"]:
                imu_file = self.contents.loc["imu", "path"]
                self._imu = NeonIMU(imu_file)
            else:
                warnings.warn("IMU data not loaded because no file was found.")
        return self._imu

    @property
    def eye_states(self) -> Optional[NeonEyeStates]:
        """
        Returns a NeonEyeStates object or None if no eye states data is found.
        """
        if self._eye_states is None:
            if self.contents.loc["3d_eye_states", "exist"]:
                eye_states_file = self.contents.loc["3d_eye_states", "path"]
                self._eye_states = NeonEyeStates(eye_states_file)
            else:
                warnings.warn(
                    "3D eye states data not loaded because no file was found."
                )
        return self._eye_states

    @property
    def blinks(self) -> Optional[NeonBlinks]:
        """
        Returns a NeonBlinks object or None if no blinks data is found.
        """
        if self._blinks is None:
            if self.contents.loc["blinks", "exist"]:
                blinks_file = self.contents.loc["blinks", "path"]
                self._blinks = NeonBlinks(blinks_file)
            else:
                warnings.warn("Blinks data not loaded because no file was found.")
        return self._blinks

    @property
    def fixations(self) -> Optional[NeonFixations]:
        """
        Returns a NeonFixations object or None if no fixations data is found.
        """
        if self._fixations is None:
            if self.contents.loc["fixations", "exist"]:
                fixations_file = self.contents.loc["fixations", "path"]
                self._fixations = NeonFixations(fixations_file)
            else:
                warnings.warn("Fixations data not loaded because no file was found.")
        return self._fixations

    @property
    def saccades(self) -> Optional[NeonSaccades]:
        """
        Returns a NeonSaccades object or None if no saccades data is found.
        """
        if self._saccades is None:
            if self.contents.loc["saccades", "exist"]:
                saccades_file = self.contents.loc["saccades", "path"]
                self._saccades = NeonSaccades(saccades_file)
            else:
                warnings.warn("Saccades data not loaded because no file was found.")
        return self._saccades

    @property
    def events(self) -> Optional[NeonEvents]:
        """
        Returns a NeonEvents object or None if no events data is found.
        """
        if self._events is None:
            if self.contents.loc["events", "exist"]:
                events_file = self.contents.loc["events", "path"]
                self._events = NeonEvents(events_file)
            else:
                warnings.warn("Events data not loaded because no file was found.")
        return self._events

    @property
    def video(self) -> Optional[NeonVideo]:
        """
        Returns a NeonVideo object or None if no scene video is found.
        """
        if self._video is None:
            if (
                (video_file := self.contents.loc["scene_video", "path"])
                and (timestamp_file := self.contents.loc["world_timestamps", "path"])
                and (video_info_file := self.contents.loc["scene_video_info", "path"])
            ):
                self._video = NeonVideo(video_file, timestamp_file, video_info_file)
            else:
                warnings.warn(
                    "Scene video not loaded because not all video-related files "
                    "(video, scene_camera.json, world_timestamps.csv) are found."
                )
        return self._video

    def concat_streams(
        self,
        stream_names: str | list[str],
        sampling_freq: Number | str = "min",
        interp_float_kind: str = "cubic",
        interp_other_kind: str = "nearest",
        inplace: bool = False,
    ) -> CustomStream:
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
            if``True``. Defaults to ``False``.

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
        return CustomStream(new_data)

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
        rec : NeonRecording
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
        inplace: bool = False,
    ) -> Optional[NeonGaze]:
        """
        Synchronize gaze data to video frames by applying windowed averaging
        around timestamps of each video frame.
        See :meth:`pyneon.stream.NeonStream.window_average` for details.

        Parameters
        ----------
        window_size : int, optional
            The size of the time window (in nanoseconds)
            over which to compute the average around each new timestamp.
            If ``None`` (default), the window size is set to the median interval
            between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
            The window size must be larger than the median interval between the original data timestamps,
            i.e., ``window_size > np.median(np.diff(data.index))``.
        inplace : bool, optional
            Whether to replace the gaze data in the instance with the window averaged data.

        Returns
        -------
        NeonGaze
            Gaze object containing data synchronized to video frames.
        """
        if self.gaze is None or self.video is None:
            raise ValueError("Gaze-video synchronization requires gaze and video data.")

        new_gaze = self.gaze.window_average(self.video.ts, window_size, inplace)
        return new_gaze

    def estimate_scanpath(
        self,
        sync_gaze: Optional["NeonGaze"] = None,
        lk_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Map fixations to video frames.

        Parameters
        ----------
        sync_gaze : NeonGaze
            Gaze data synchronized to video frames. If None (default),
            a windowed average is applied to synchronize gaze data to video frames.
        lk_params : dict
            Parameters for the Lucas-Kanade optical flow algorithm.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the scanpath with updated fixation points.
        """
        if sync_gaze is None:
            sync_gaze = self.sync_gaze_to_video()
        if (video := self.video) is None:
            raise ValueError("Estimating scanpath requires video data.")
        return estimate_scanpath(video, sync_gaze, lk_params)

    def detect_apriltags(self, tag_family: str = "tag36h11") -> pd.DataFrame:
        """
        Detect AprilTags in a video and report their data for every frame using the apriltag library.

        Parameters
        ----------
        tag_family : str, optional
            The AprilTag family to detect (default is 'tag36h11').

        Returns
        -------
        pd.DataFrame
            A DataFrame containing AprilTag detections, with columns:
            - 'frame_idx': The frame number
            - 'tag_id': The ID of the detected AprilTag
            - 'corners': A 4x2 array of the tag corner coordinates
            - 'center': A 1x2 array with the tag center coordinates
        """
        # Check if JSON already exists
        if (json_file := self.der_dir / "apriltags.json").is_file():
            return pd.read_json(json_file, orient="records", lines=True)

        all_detections = detect_apriltags(self.video, tag_family)
        # Save to JSON
        all_detections.to_json(
            self.der_dir / "apriltags.json", orient="records", lines=True
        )

        return all_detections

    def estimate_camera_pose(
        self,
        tag_locations_df: pd.DataFrame,
        all_detections: pd.DataFrame = pd.DataFrame(),
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
            If empty, the detections are computed using the `detect_apriltags` function.

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

        # check for detections dataframe
        if all_detections.empty:
            detection_file = self.der_dir / "apriltags.json"
            # open apriltags
            if detection_file.is_file():
                all_detections = pd.read_json(
                    detection_file, orient="records", lines=True
                )
            else:
                all_detections = self.detect_apriltags()

        # Check if result JSON already exists
        json_file = self.der_dir / "camera_pose.json"
        if json_file.is_file() and not overwrite:
            return pd.read_json(json_file, orient="records")

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
                camera_pose_raw = pd.read_json(json_file, orient="records")
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
