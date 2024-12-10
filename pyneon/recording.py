from pathlib import Path
from typing import Union, Literal, Optional, Dict, List
import pandas as pd
import numpy as np
import json
from datetime import datetime
import copy
import warnings
import matplotlib.pyplot as plt
from numbers import Number

from .stream import NeonGaze, NeonIMU, NeonEyeStates, CustomStream
from .events import NeonBlinks, NeonFixations, NeonSaccades, NeonEvents
from .preprocess import concat_streams, concat_events, smooth_camera_positions
from .video import (
    NeonVideo,
    sync_gaze_to_video,
    estimate_scanpath,
    detect_apriltags,
    compute_camera_positions,
)
from .vis import plot_distribution, plot_scanpath_on_video
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
        ├── <scene_video>.mp4 (if present)
        ├── scanpath.pkl (after executing `estimate_scanpath`)
        └── video_with_scanpath.mp4 (after executing `overlay_scanpath_on_video`)

    Streams, events, (and scene video) will be located but not loaded until
    accessed as properties such as ``gaze``, ``imu``, and ``eye_states``.

    Parameters
    ----------
    recording_dir : str or :class:`pathlib.Path`
        Path to the directory containing the recording.

    Attributes
    ----------
    recording_id : str
        Recording ID.
    recording_dir : :class:`pathlib.Path`
        Path to the recording directory.
    info : dict
        Information about the recording. Read from ``info.json``. For details, see
        https://docs.pupil-labs.com/neon/data-collection/data-format/#info-json.
    start_time : int
        Start time (in ns) of the recording as in ``info.json``.
        May not match the start time of each data stream.
    start_datetime : :class:`datetime.datetime`
        Start time (datetime) of the recording as in ``info.json``.
        May not match the start time of each data stream.
    contents : :class:`pandas.DataFrame`
        Contents of the recording directory. Each index is a stream or event name
        (e.g. ``gaze`` or ``imu``) and columns are ``exist`` (bool),
        ``filename`` (str), and ``path`` (Path).
    """

    def __init__(self, recording_dir: Union[str, Path]):
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
Wearer ID: {self.info['wearer_id']}
Wearer name: {self.info['wearer_name']}
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
        stream_names: Union[str, list[str]],
        sampling_freq: Union[Number, str] = "min",
        resamp_float_kind: str = "linear",
        resamp_other_kind: str = "nearest",
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
            Sampling frequency to resample the streams to.
            If numeric, the streams will be resampled to this frequency.
            If ``"min"``, the lowest nominal sampling frequency
            of the selected streams will be used.
            If ``"max"``, the highest nominal sampling frequency will be used.
            Defaults to ``"min"``.
        resamp_float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
        resamp_other_kind : str, optional
            Kind of interpolation applied on columns of other types.
            Defaults to ``"nearest"``.
        inplace : bool, optional
            Replace selected stream data with resampled data during concatenation
            if``True``. Defaults to ``False``.

        Returns
        -------
        concat_data : :class:`pandas.DataFrame`
            Concatenated data.
        """
        new_data = concat_streams(
            self,
            stream_names,
            sampling_freq,
            resamp_float_kind,
            resamp_other_kind,
            inplace,
        )
        return CustomStream(new_data)

    def concat_events(self, event_names: Union[str, list[str]]) -> pd.DataFrame:
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
        concat_events : :class:`pandas.DataFrame`
            Concatenated events.
        """
        return concat_events(self, event_names)

    def plot_distribution(
        self,
        heatmap_source: Literal["gaze", "fixations", None] = "gaze",
        scatter_source: Literal["gaze", "fixations", None] = "fixations",
        step_size: int = 10,
        sigma: Union[float, None] = 2,
        width_height: tuple[int, int] = (1600, 1200),
        cmap: Union[str, None] = "inferno",
        ax: Union[plt.Axes, None] = None,
        show: bool = True,
    ):
        """
        Plot a heatmap of gaze or fixation data on a matplotlib axis.
        Users can flexibly choose to generate a smoothed heatmap and/or scatter plot and
        the source of the data (gaze or fixation).

        Parameters
        ----------
        rec : :class:`NeonRecording`
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
        sigma : float or None
            Standard deviation of the Gaussian kernel used to smooth the heatmap.
            If None or 0, no smoothing is applied. Defaults to 2.
        width_height : tuple[int, int]
            If video is not available, the width and height of the scene camera frames to
            specify the heatmap dimensions. Defaults to (1600, 1200).
        cmap : str or None
            Colormap to use for the heatmap. Defaults to 'inferno'.
        ax : :class:`matplotlib.pyplot.Axes` or None
            Axis to plot the frame on. If ``None``, a new figure is created.
            Defaults to ``None``.
        show : bool
            Show the figure if ``True``. Defaults to True.

        Returns
        -------
        fig : :class:`matplotlib.pyplot.Figure`
            Figure object containing the plot.
        ax : :class:`matplotlib.pyplot.Axes`
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
    ) -> NeonGaze:
        """
        Synchronize gaze data to video frames by applying windowed averaging
        around each video frame timestamp.

        Parameters:
        -----------
        rec : NeonRecording
            Recording object containing gaze and video data.
        window_size : int, optional
            The size of the time window (in nanoseconds)
            over which to compute the average around each new timestamp.
            If ``None`` (default), the window size is set to the median interval
            between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
            The window size must be larger than the median interval between the original data timestamps,
            i.e., ``window_size > np.median(np.diff(data.index))``.

        Returns
        -------
        NeonGaze
            Gaze object containing data synchronized to video frames.
        """
        new_data = sync_gaze_to_video(self, window_size)
        new_gaze = copy.copy(self.gaze)
        new_gaze.data = new_data
        return new_gaze

    def estimate_scanpath(
        self,
        sync_gaze: Optional["NeonGaze"] = None,
        lk_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Map fixations to video frames.

        Parameters:
        -----------
        sync_gaze : NeonGaze
            Gaze data synchronized to video frames. If None (default),
            a windowed average is applied to synchronize gaze data to video frames.
        lk_params : dict
            Parameters for the Lucas-Kanade optical flow algorithm.
        """
        if sync_gaze is None:
            sync_gaze = self.sync_gaze_to_video()
        if (video := self.video) is None:
            raise ValueError("Estimating scanpath requires video data.")
        return estimate_scanpath(video, sync_gaze, lk_params)
    
    def detect_apriltags(
        self,
        tag_family: str ='tag36h11'
    ) -> pd.DataFrame:
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
        if (json_file := self.recording_dir / "apriltags.json").is_file():
            return pd.read_json(json_file, orient="records", lines=True)

        all_detections = detect_apriltags(self.video, tag_family)
        # Save to JSON
        all_detections.to_json(self.recording_dir / "apriltags.json", orient="records", lines=True)

        return all_detections


    def compute_camera_positions(
        self,
        tag_locations: Dict[int, List[float]],
        tag_size: Union[float, Dict[int, float]],
        all_detections: pd.DataFrame = pd.DataFrame(),
    ) -> pd.DataFrame:
        """
        Compute the camera positions from AprilTag detections in a video.

        Parameters
        ----------
        tag_locations : dict
            A dictionary mapping AprilTag IDs to their 3D locations in the scene.
        tag_size : float or dict
            The size of the AprilTags in the scene. If a float, all tags are assumed to have the same size.
            If a dictionary, the keys are the AprilTag IDs and the values are the sizes.
        all_detections : pd.DataFrame, optional
            DataFrame containing the AprilTag detections for each frame, with columns:
            - 'frame_idx': The frame number
            - 'tag_id': The ID of the detected AprilTag
            - 'corners': A 4x2 array of the tag corner coordinates
            - 'center': A 1x2 array with the tag center coordinates
            If None, the detections are computed using the `detect_apriltags` function.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the camera positions for each frame, with columns:
            - 'frame_idx': The frame number
            - 'position': A 1x3 array with the camera position
            - 'rotation': A 1x3 array with the camera rotation
        """

        # Check if JSON already exists
        if (json_file := self.recording_dir / "camera_positions.json").is_file():
            return pd.read_json(json_file, orient="records")

        camera_positions = compute_camera_positions(self.video, tag_locations, tag_size, all_detections)
        # Save to JSON
        camera_positions.to_json(self.recording_dir / "camera_positions.json", orient="records")

        return camera_positions


    def smooth_camera_positions(
        self,
        camera_position_raw: pd.DataFrame = pd.DataFrame(),
        state_dim: int = 3,
        meas_dim: int = 3,
        process_noise: float = 0.005,
        measurement_noise: float = 0.005,
        gating_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Apply a Kalman filter to smooth camera positions and gate outliers based on Mahalanobis distance.
        Expects a DataFrame containing 'frame_idx' and 'camera_pos' columns, where 'camera_pos' is a
        length-3 array-like object representing [x, y, z] coordinates.

        Parameters
        ----------
        camera_position_raw : pd.DataFrame
            DataFrame containing 'frame_idx' and 'camera_pos' columns.
        state_dim : int, optional
            Dimensionality of the state vector. Default is 3 (x, y, z).
        meas_dim : int, optional
            Dimensionality of the measurement vector. Default is 3 (x, y, z).
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

        if camera_position_raw.empty:
            # Check if JSON already exists
            if (json_file := self.recording_dir / "camera_positions.json").is_file():
                camera_position_raw = pd.read_json(json_file, orient="records")
                # Ensure 'camera_pos' is parsed as NumPy arrays
                camera_position_raw['camera_pos'] = camera_position_raw['camera_pos'].apply(
                    lambda pos: np.array(pos, dtype=float)
                )
            else:
                # Run the function to get the data
                camera_position_raw = self.compute_camera_positions()

        smoothed_positions = smooth_camera_positions(
            camera_position_raw,
            state_dim,
            meas_dim,
            process_noise,
            measurement_noise,
            gating_threshold
        )

        # Save to JSON
        smoothed_positions.to_json(self.recording_dir / "camera_positions.json", orient="records")

        return smoothed_positions


    def plot_scanpath_on_video(
        self,
        scanpath: pd.DataFrame,
        circle_radius: int = 10,
        line_thickness: Optional[int] = 2,
        max_fixations: int = 10,
        show_video: bool = False,
        video_output_path: Optional[Union[Path, str]] = "scanpath.mp4",
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
        if (video := self.video) is None:
            raise ValueError("Estimating scanpath requires video data.")
        plot_scanpath_on_video(
            video,
            scanpath,
            circle_radius,
            line_thickness,
            max_fixations,
            show_video,
            video_output_path,
        )

    def export_motion_bids(
        self,
        motion_dir: Union[str, Path],
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
        motion_dir : str or :class:`pathlib.Path`
            Output directory to save the Motion-BIDS formatted data.
        prefix : str, optional
            Prefix for the BIDS filenames, by default "sub-XX_task-YY_tracksys-NeonIMU".
            The format should be `sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]`
            (Fields in [] are optional). Files will be saved as
            ``{prefix}_motion.<tsv|json>``.

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
        output_dir: Union[str, Path],
        prefix: str = "",
        extra_metadata: dict = {},
    ):
        """
        Export eye-tracking data to Eye-tracking-BIDS format. Continuous samples
        and events are saved to .tsv.gz files with accompanying .json metadata files.
        Users should later edit the metadata files according to the experiment.

        Parameters
        ----------

        output_dir : str or :class:`pathlib.Path`
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
