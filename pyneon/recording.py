from pathlib import Path
from typing import Union
import pandas as pd
import json
from datetime import datetime

from .data import NeonGaze, NeonIMU, NeonEyeStates
from .preprocess import concat_channels
from .io import export_motion_bids, exports_eye_bids


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
        └── <scene_video>.mp4 (if present)

    Channels, events, (and scene video) will be located but not loaded until
    accessed as attributes such as ``gaze``, ``imu``, and ``eye_states``.

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
    start_datetime : :class:`datetime.datetime`
        Start time of the recording as in ``info.json``.
        May not match the start time of each data channel.
    contents : :class:`pandas.DataFrame`
        Contents of the recording directory. Each index is a channel or event name
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
        self.start_datetime = datetime.fromtimestamp(self.info["start_time"] / 1e9)

        self.recording_id = self.info["recording_id"]
        self.recording_dir = recording_dir

        self._eye_states = None
        self._blinks = None
        self._events = None
        self._fixations = None
        self._gaze = None
        self._imu = None

        self._get_contents()

    def __repr__(self) -> str:
        return f"""
Recording ID: {self.recording_id}
Wearer ID: {self.info['wearer_id']}
Wearer name: {self.info['wearer_name']}
Recording start time: {self.start_datetime}
Recording duration: {self.info["duration"] / 1e9} s
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
                with open(camera_info) as f:
                    self.camera_info = json.load(f)
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
    def gaze(self) -> NeonGaze:
        """
        Returns a NeonGaze object or None if no gaze data is found.
        """
        if self._gaze is None:
            if self.contents.loc["gaze", "exist"]:
                gaze_file = self.contents.loc["gaze", "path"]
                self._gaze = NeonGaze(gaze_file)
            else:
                raise UserWarning(
                    "Gaze data not loaded because no recording was found."
                )
        return self._gaze

    @property
    def imu(self) -> NeonIMU:
        """
        Returns a NeonIMU object or None if no IMU data is found.
        """
        if self._imu is None:
            if self.contents.loc["imu", "exist"]:
                imu_file = self.contents.loc["imu", "path"]
                self._imu = NeonIMU(imu_file)
            else:
                raise UserWarning("IMU data not loaded because no recording was found.")
        return self._imu

    @property
    def eye_states(self) -> NeonEyeStates:
        """
        Returns a NeonEyeStates object or None if no eye states data is found.
        """
        if self._eye_states is None:
            if self.contents.loc["3d_eye_states", "exist"]:
                eye_states_file = self.contents.loc["3d_eye_states", "path"]
                self._eye_states = NeonEyeStates(eye_states_file)
            else:
                raise UserWarning(
                    "3D eye states data not loaded because no recording was found."
                )
        return self._eye_states

    def concat_channels(
        self,
        ch_names: list[str],
        sampling_freq: Union[float, int, str] = "min",
        resamp_float_kind: str = "linear",
        resamp_other_kind: str = "nearest",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Concatenate data from different channels under common timestamps.
        Since the signals may have different timestamps and sampling frequencies,
        resampling of all signals to a set of common timestamps is performed.
        The latest start timestamp and earliest last timestamp of the selected channels
        are used to define the common timestamps.

        Parameters
        ----------
        ch_names : list of str
            List of channel names to concatenate. Channel names must be in
            ``{"gaze", "imu", "eye_states", "3d_eye_states"}``.
        sampling_freq : float or int or str, optional
            Sampling frequency to resample the signals to.
            If numeric, the signals will be resampled to this frequency.
            If ``"min"``, the lowest nominal sampling frequency
            of the selected channels will be used.
            If ``"max"``, the highest nominal sampling frequency will be used.
            Defaults to ``"min"``.
        resamp_float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            Defaults to ``"linear"``. For details see :class:`scipy.interpolate.interp1d`.
        resamp_other_kind : str, optional
            Kind of interpolation applied on columns of other types.
            Defaults to ``"nearest"``.
        inplace : bool, optional
            Replace selected signal data with resampled data during concatenation\
            if ``True``. Defaults to ``False``.

        Returns
        -------
        concat_data : :class:`pandas.DataFrame`
            Concatenated data.
        """
        return concat_channels(
            self, ch_names, sampling_freq, resamp_float_kind, resamp_other_kind, inplace
        )

    def to_motion_bids(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sub-XX_task-YY_tracksys-NeonIMU",
    ):
        """
        Export IMU data to Motion-BIDS format. Continuous samples are saved to a .tsv
        file and metadata (with template fields) are saved to a .json file.
        Users should later edit the metadata file according to the experiment to make
        it BIDS-compliant.

        Parameters
        ----------
        output_dir : str or :class:`pathlib.Path`
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
        export_motion_bids(self, output_dir)

    def to_eye_bids(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sub-XX_task-YY_tracksys-NeonGaze",
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
        exports_eye_bids(self, output_dir)
