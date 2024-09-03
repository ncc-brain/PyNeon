from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import json

from .data import NeonGaze, NeonIMU, NeonEyeStates
from .preprocess import concat_channels


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
    located but not loaded until accessed as attributes such as ``gaze``, ``imu``,
    and ``eye_states``.
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
    def gaze(self):
        """
        Returns a `NeonGaze` object.
        Loads gaze data if not already loaded.
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
    def imu(self):
        """
        Returns a `NeonIMU` object.
        Loads IMU data if not already loaded.
        """
        if self._imu is None:
            if self.contents.loc["imu", "exist"]:
                imu_file = self.contents.loc["imu", "path"]
                self._imu = NeonIMU(imu_file)
            else:
                raise UserWarning("IMU data not loaded because no recording was found.")
        return self._imu

    @property
    def eye_states(self):
        """
        Returns a `NeonEyeStates` object.
        Loads 3D eye states data if not already loaded.
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
        ch_names : list[str]
            List of channel names to concatenate. Channel names must be one of
            {"gaze", "imu", "eye_states", "3d_eye_states"}.
        downsample : bool, optional
            If True, downsample the signals to the lowest sampling rate of the selected
            channels. If False, the signals will be resampled to the highest sampling
            rate. By default True.
        resamp_float_kind : str, optional
            Resampling method for columns of float type, by default "linear".
        resamp_other_kind : str, optional
            Resampling method for columns of other types, by default "nearest".

        Returns
        -------
        concat_data : DataFrame
            Concatenated data.
        """
        return concat_channels(self, ch_names, resamp_float_kind, resamp_other_kind)
