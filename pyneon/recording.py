import json
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
from .export import export_cloud_format, export_eye_tracking_bids, export_motion_bids
from .preprocess import concat_events, concat_streams
from .stream import Stream
from .utils.doc_decorators import fill_doc
from .utils.variables import calib_dtype, expected_files_cloud, expected_files_native
from .video import Video
from .vis import plot_distribution


class Recording:
    """
    Container of a multi-modal recording with
    :class:`Stream`, :class:`Events`, and :class:`Video`

    The recording directory is expected to follow either the Pupil Cloud format
    (tested with data format version >= 2.3) or the native Pupil Labs format
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
    accessed as properties such as :attr:`gaze`, :attr:`fixations`, and :attr:`scene_video`.

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
    format : {"cloud", "native"}
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
        self.data_format_version = info.get("data_format_version", "unknown")

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
Data format: {self.format} (version: {self.data_format_version})
Recording ID: {self.recording_id}
Wearer ID: {self.info.get("wearer_id", "N/A")}
Wearer name: {self.info.get("wearer_name", "N/A")}
Recording start time: {self.start_datetime}
Recording duration: {self.info["duration"]} ns ({self.info["duration"] / 1e9} s)
"""

    def close(self) -> None:
        """Release cached video handles, if any."""
        for attr_name in ("scene_video", "eye_video"):
            video = self.__dict__.get(attr_name)
            # Check if video is a Video instance and not None
            if video is not None and hasattr(video, "release"):
                try:
                    video.close()
                except Exception:
                    pass
                self.__dict__[attr_name] = None

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def _load_property(self, property_name: str, loader):
        try:
            return loader()
        except Exception as e:
            message = f"Recording.{property_name} cannot be read because: {e}"
            raise ValueError(message) from e

    @cached_property
    def gaze(self) -> Stream:
        """
        Return a cached :class:`Stream` instance containing gaze data.

        For **Pupil Cloud** recordings, the data is loaded from ``gaze.csv``.

        For **native** recordings, the data is loaded from ``gaze_200hz.raw`` (if present;
        otherwise from ``gaze ps1.raw``) along with the corresponding ``.time`` and
        ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                gaze_200hz_file = self.recording_dir / "gaze_200hz.raw"
                if gaze_200hz_file.is_file():
                    return Stream(gaze_200hz_file)
                return Stream(self.recording_dir / "gaze ps1.raw")
            return Stream(self.recording_dir / "gaze.csv")

        return self._load_property("gaze", _load)

    @cached_property
    def imu(self) -> Stream:
        """
        Return a cached :class:`Stream` instance containing IMU data.

        For **Pupil Cloud** recordings, the data is loaded from ``imu.csv``.

        For **native** recordings, the data is loaded from ``imu ps1.raw``, along with
        the corresponding ``.time`` and ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                return Stream(self.recording_dir / "imu ps1.raw")
            return Stream(self.recording_dir / "imu.csv")

        return self._load_property("imu", _load)

    @cached_property
    def eye_states(self) -> Stream:
        """
        Return a cached :class:`Stream` instance containing eye states data.

        For **Pupil Cloud** recordings, the data is loaded from ``3d_eye_states.csv``.

        For **native** recordings, the data is loaded from ``eye_state ps1.raw``, along
        with the corresponding ``.time`` and ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                return Stream(self.recording_dir / "eye_state ps1.raw")
            return Stream(self.recording_dir / "3d_eye_states.csv")

        return self._load_property("eye_states", _load)

    @cached_property
    def blinks(self) -> Events:
        """
        Return a cached :class:`Events` instance containing blink event data.

        For **Pupil Cloud** recordings, the data is loaded from ``blinks.csv``.

        For **native** recordings, the data is loaded from ``blinks ps1.raw``, along with
        the corresponding ``.time`` and ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                return Events(self.recording_dir / "blinks ps1.raw")
            return Events(self.recording_dir / "blinks.csv")

        return self._load_property("blinks", _load)

    @cached_property
    def fixations(self) -> Events:
        """
        Return a cached :class:`Events` instance containing fixations data.

        For **Pupil Cloud** recordings, the data is loaded from ``fixations.csv``.

        For **native** recordings, the data is loaded from ``fixations ps1.raw``,
        along with the corresponding ``.time`` and ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                return Events(self.recording_dir / "fixations ps1.raw", "fixations")
            return Events(self.recording_dir / "fixations.csv")

        return self._load_property("fixations", _load)

    @cached_property
    def saccades(self) -> Events:
        """
        Return a cached :class:`Events` instance containing saccades data.

        For **Pupil Cloud** recordings, the data is loaded from ``saccades.csv``.

        For **native** recordings, the data is loaded from ``fixations ps1.raw``,
        along with the corresponding ``.time`` and ``.dtype`` files.
        """

        def _load():
            if self.format == "native":
                # Note: In the native format, both fixations and saccades are stored in 'fixations ps1.raw'.
                # The event type argument ("saccades") is used to distinguish which events to load.
                return Events(self.recording_dir / "fixations ps1.raw", "saccades")
            return Events(self.recording_dir / "saccades.csv")

        return self._load_property("saccades", _load)

    @cached_property
    def events(self) -> Events:
        """
        Return a cached :class:`Events` instance containing events data.

        For **Pupil Cloud** recordings, the events data is loaded from ``events.csv``.

        For **native** recordings, the events data is loaded from ``event.txt`` and
        ``event.time``.
        """

        def _load():
            if self.format == "native":
                return Events(self.recording_dir / "event.txt")
            return Events(self.recording_dir / "events.csv")

        return self._load_property("events", _load)

    @cached_property
    def scene_video(self) -> Video:
        """
        Return a cached :class:`Video` instance containing scene video data.

        For **Pupil Cloud** recordings, the video is loaded from the only ``*.mp4`` file
        in the recording directory.

        For **native** recordings, the video is loaded from the ``Neon Scene Camera*.mp4``
        file in the recording directory.
        """

        def _load():
            reg_exp = "Neon Scene Camera*.mp4" if self.format == "native" else "*.mp4"
            video_files = list(self.recording_dir.glob(reg_exp))
            if len(video_files) == 0:
                raise FileNotFoundError(
                    f"No scene video file found in {self.recording_dir}"
                )
            if len(video_files) > 1:
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

        return self._load_property("scene_video", _load)

    @cached_property
    def eye_video(self) -> Video:
        """
        Return a cached :class:`Video` instance containing eye video data.

        Eye video is only available for **native** recordings and is loaded from the
        ``Neon Sensor Module*.mp4`` file in the recording directory.
        """

        def _load():
            if self.format == "cloud":
                raise ValueError("Pupil Cloud recordings do not contain eye video.")
            video_files = list(self.recording_dir.glob("Neon Sensor Module*.mp4"))
            if len(video_files) == 0:
                raise FileNotFoundError(
                    f"No eye video file found in {self.recording_dir}"
                )
            if len(video_files) > 1:
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
            return Video(video_file, ts, None)

        return self._load_property("eye_video", _load)

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
        The latest start timestamp and earliest last timestamp of the selected streams
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
            Stream instance containing concatenated data.
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

    def concat_events(self, events_names: str | list[str]) -> Events:
        """
        Concatenate different events. All columns in the selected event type will be
        present in the final DataFrame. An additional "type" column denotes the event
        type. If ``events`` is selected, its "timestamp [ns]" column will be
        renamed to "start timestamp [ns]", and the "name" and "type" columns will
        be renamed to "message name" and "message type" respectively to provide
        a more readable output.

        Parameters
        ----------
        events_names : list of str
            List of event names to concatenate. Event names must be in
            ``{"blinks", "fixations", "saccades", "events"}``
            (singular forms are tolerated).

        Returns
        -------
        Events
            Events instance containing concatenated data.
        """
        new_data = concat_events(self, events_names)
        return Events(new_data)

    @fill_doc
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
        %(ax_param)s
        %(show_param)s

        Returns
        -------
        %(fig_ax_return)s
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
    ) -> Optional[Stream]:
        """
        Synchronize gaze data to video frames by applying windowed averaging
        around timestamps of each video frame.

        Parameters
        ----------
        window_size : int, optional
            Size of the time window in nanoseconds used for averaging.
            If None, defaults to the median interval between video frame timestamps.
        inplace : bool, optional
            If True, update the gaze stream in-place and return None.
            If False, return a new Stream. Defaults to False.

        Returns
        -------
        Stream or None
            A Stream indexed by "timestamp [ns]" containing the window-averaged
            gaze data, or None if ``inplace=True``.
        """
        gaze = self.gaze
        video = self.scene_video
        return gaze.window_average(video.ts, window_size, inplace=inplace)

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
        Export IMU data to Motion-BIDS format.

        Motion-BIDS [1]_ is an extension to the Brain Imaging Data Structure (BIDS) that
        standardizes motion sensor data from IMUs for reproducible research. This method
        creates motion time-series and metadata files, channels files, and updates the
        scans file in the subject/session directory. The output files are BIDS-compliant
        templates and may require additional metadata editing for full compliance with
        your specific use case.

        For instance, an exported directory could look like this:

        .. code-block:: text

            sub-01/
                ses-1/
                    motion/
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_channels.json
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_channels.tsv
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_motion.json
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_motion.tsv
                    sub-01_ses-1_scans.tsv

        Motion-BIDS specification can be found at:
        https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html

        Parameters
        ----------
        motion_dir : str or pathlib.Path
            Output directory to save the Motion-BIDS formatted data.
        prefix : str, optional
            BIDS naming prefix. The format is:

            ``sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]``

            Required fields are ``sub-<label>``, ``task-<label>``, and
            ``tracksys-<label>`` (use ``tracksys-NeonIMU`` for Neon IMU data).
            Files are saved as ``{prefix}_motion.<tsv|json>`` and
            ``{prefix}_channels.<tsv|json>``.
        extra_metadata : dict, optional
            Extra metadata to include in the JSON metadata file. Keys must be valid
            BIDS fields (for example, ``TaskName``).

        References
        ----------
        .. [1] Jeung, S., Cockx, H., Appelhoff, S., Berg, T., Gramann, K., Grothkopp, S., Warmerdam, E., Hansen, C., Oostenveld, R., & Welzel, J. (2024). Motion-BIDS: An extension to the brain imaging data structure to organize motion data for reproducible research. Scientific Data, 11(1), 716. https://doi.org/10.1038/s41597-024-03559-8
        """
        export_motion_bids(self, motion_dir, prefix, extra_metadata)

    def export_eye_tracking_bids(
        self,
        output_dir: str | Path,
        prefix: Optional[str] = None,
        extra_metadata: dict = {},
    ):
        """
        Export eye-tracking data to Eye-Tracking-BIDS format.

        Eye-Tracking-BIDS [2]_ standardizes gaze position, pupil data, and eye-tracking
        events by treating eye-tracking data as physiological data that can be organized
        alongside most BIDS modalities. The export creates the following files:

        - Gaze coordinates (x, y) as time-series data (``_physio.tsv.gz``), including
          pupil diameter measurements when available
        - Eye events (blinks, saccades, fixations) and experimental messages as event
          markers (``_physioevents.tsv.gz``)

        Accompanying metadata is saved in JSON sidecar files.

        An example exported directory could look like this:

        .. code-block:: text

            sub-01/
                ses-1/
                    motion/
                        <existing motion files if motion export is performed>
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_physio.json
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_physio.tsv.gz
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_physioevents.json
                        sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1_physioevents.tsv.gz
                    sub-01_ses-1_scans.tsv

        BIDS specifications for physiological recordings and specifically eye-tracking data can be found at:

        - https://bids-specification.readthedocs.io/en/stable/modality-specific-files/physiological-recordings.html
        - https://bids-specification.readthedocs.io/en/stable/modality-specific-files/physiological-recordings.html#eye-tracking

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Output directory to save the Eye-tracking-BIDS formatted data.
        prefix : str, optional
            BIDS naming prefix. Must include ``sub-<label>`` at minimum. Files are
            saved as ``{prefix}_physio.<tsv.gz|json>`` and
            ``{prefix}_physioevents.<tsv.gz|json>``.
        extra_metadata : dict, optional
            Extra metadata to include in the JSON metadata file. Keys must be valid
            BIDS fields.

        References
        ----------
        .. [2] Szinte, M., Bach, D. R., Draschkow, D., Esteban, O., Gagl, B., Gau, R., Gregorova, K., Halchenko, Y. O., Huberty, S., Kling, S. M., Kulkarni, S., Maintainers, T. B., Markiewicz, C. J., Mikkelsen, M., Oostenveld, R., & Pfarr, J.-K. (2026). Eye-Tracking-BIDS: The Brain Imaging Data Structure extended to gaze position and pupil data. bioRxiv. https://doi.org/10.64898/2026.02.03.703514
        """
        export_eye_tracking_bids(self, output_dir, prefix, extra_metadata)
