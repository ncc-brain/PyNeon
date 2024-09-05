from pathlib import Path
from typing import Union
import pandas as pd
import json

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
    must contain an ``info.json`` file. Channels, events, (and scene video) will be
    located but not loaded until accessed as  attributes such as ``gaze``, ``imu``,
    and ``eye_states``.

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
        Information about the recording. Read from ``info.json``.
    contents : :class:`pandas.DataFrame`
        DataFrame containing the contents of the recording directory.
    """

    def __init__(self, recording_dir: Union[str, Path]):
        recording_dir = Path(recording_dir)
        if not recording_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {recording_dir}")

        if (info_path := recording_dir / "info.json").is_file():
            with open(info_path) as f:
                self.info = json.load(f)
        else:
            raise FileNotFoundError(f"info.json not found in {recording_dir}")

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
        contents_to_print = self.contents.drop(columns="path", inplace=False)
        return f"NeonRecording | {self.recording_id}\n{contents_to_print.to_string()}"

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

    def load(self):
        self.gaze
        self.imu
        self.eye_states

    def concat_channels(
        self,
        ch_names: list[str],
        downsample: bool = True,
        resamp_float_kind: str = "linear",
        resamp_other_kind: str = "nearest",
    ) -> pd.DataFrame:
        """
        Returns a single pandas.DataFrame under common timestamps.
        This will require interpolation of all signals to the same timestamps. See
        `pyneon.preprocess.concat_channels` for more details.

        Parameters
        ----------
        ch_names : list of str
            List of channel names to concatenate. Channel names must be one of
            {"gaze", "imu", "eye_states", "3d_eye_states"}.
        downsample : bool, optional
            If True, downsample the signals to the lowest sampling rate of the selected
            channels. If False, the signals will be resampled to the highest sampling
            rate. By default True.
        resamp_float_kind : str, optional
            The kind of interpolation applied on columns of float type,
            by default "linear". See `scipy.interpolate.interp1d` for more details.
        resamp_other_kind : str, optional
            The kind of interpolation applied on columns of other types,
            by default "nearest".

        Returns
        -------
        concat_data : :class:`pandas.DataFrame`
            Concatenated data.
        """
        return concat_channels(
            self, ch_names, downsample, resamp_float_kind, resamp_other_kind
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
