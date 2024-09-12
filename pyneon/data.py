from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union
import cv2

from .preprocess import resample


class NeonData:
    """
    Base for Neon tabular data. It reads from a CSV file and stores the data
    as a pandas DataFrame (with section and recording IDs removed).
    """

    def __init__(self, file: Path):
        self.file = file
        if isinstance(file, Path) and file.suffix == ".csv":
            data = pd.read_csv(file)
        else:  # TODO: Implement reading native data formats
            pass

        if data["recording id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple recording IDs")
        self.data = data.drop(columns=["recording id"])

        # Every data file except events.csv has a section id column
        if "section id" in self.data.columns:
            if data["section id"].nunique() > 1:
                raise ValueError(f"{file.name} contains multiple section IDs")
            self.data.drop(columns=["section id"], inplace=True)

    def __len__(self) -> int:
        return self.data.shape[0]


class NeonStream(NeonData):
    """
    Base for Neon continuous data (gaze, eye states, IMU).
    It must contain a ``timestamp [ns]`` column.

    Parameters
    ----------
    file : :class:`pathlib.Path`
        Path to the CSV file containing the stream data.

    Attributes
    ----------
    data : pandas.DataFrame
        DataFrame containing the stream data.
    timestamps : np.ndarray
        Timestamps of the stream.
    ts : np.ndarray
        Alias for timestamps.
    first_ts : int
        First timestamp of the stream.
    last_ts : int
        Last timestamp of the stream.
    times : np.ndarray
        Timestamps converted to seconds relative to stream start.
    duration : float
        Duration of the stream in seconds.
    sampling_freq_effective : float
        Effective sampling frequency of the stream
        (number of time points divided by duration).
    sampling_freq_nominal : int
        Nominal sampling frequency of the stream as specified by Pupil Labs
        (https://pupil-labs.com/products/neon/specs).
    """

    def __init__(self, file: Path):
        super().__init__(file)
        self._get_attributes()

    def _get_attributes(self):
        """
        Get attributes given self.data DataFrame.
        """
        self.data.sort_values(by=["timestamp [ns]"], inplace=True)
        self.timestamps = self.data["timestamp [ns]"].to_numpy()
        self.ts = self.timestamps
        self.first_ts = int(self.ts[0])
        self.last_ts = int(self.ts[-1])
        self.times = (self.ts - self.first_ts) / 1e9
        self.data["time [s]"] = self.times
        self.duration = float(self.times[-1] - self.times[0])
        self.sampling_freq_effective = self.data.shape[0] / self.duration

    def resample(
        self,
        new_ts: Union[None, np.ndarray] = None,
        float_kind: str = "linear",
        other_kind: str = "nearest",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Resample the stream to a new set of timestamps.

        Parameters
        ----------
        new_ts : np.ndarray, optional
            New timestamps to resample the stream to. If ``None``,
            the stream is resampled to its nominal sampling frequency according to
            https://pupil-labs.com/products/neon/specs.
        float_kind : str, optional
            Kind of interpolation applied on columns of float type,
            by default "linear". For details see :class:`scipy.interpolate.interp1d`.
        other_kind : str, optional
            Kind of interpolation applied on columns of other types,
            by default "nearest".
        inplace : bool, optional
            Replace stream data with resampled data if ``True``,
            by default ``False``.

        Returns
        -------
        pandas.DataFrame
            Resampled data.
        """
        # If new_ts is not provided, generate a evenly spaced array of timestamps
        if new_ts is None:
            step_size = int(1e9 / self.sampling_freq_nominal)
            new_ts = np.arange(self.first_ts, self.last_ts, step_size, dtype=np.int64)
            assert new_ts[0] == self.first_ts
            assert np.all(np.diff(new_ts) == step_size)
        resamp_data = resample(new_ts, self.data, float_kind, other_kind)
        if inplace:
            self.data = resamp_data
            self._get_attributes()
        return resamp_data


class NeonGaze(NeonStream):
    """
    Gaze data.
    """

    def __init__(self, file: Path):
        super().__init__(file)
        self.sampling_freq_nominal = int(200)
        self.data = self.data.astype(
            {
                "timestamp [ns]": "Int64",
                "gaze x [px]": float,
                "gaze y [px]": float,
                "worn": bool,
                "fixation id": "Int32",
                "blink id": "Int32",
                "azimuth [deg]": float,
                "elevation [deg]": float,
                "time [s]": float,
            }
        )


class NeonEyeStates(NeonStream):
    """
    3D eye states data.
    """

    def __init__(self, file: Path):
        super().__init__(file)
        self.sampling_freq_nominal = 200
        self.data = self.data.astype(
            {
                "timestamp [ns]": "Int64",
                "pupil diameter left [mm]": float,
                "pupil diameter right [mm]": float,
                "eyeball center left x [mm]": float,
                "eyeball center left y [mm]": float,
                "eyeball center left z [mm]": float,
                "eyeball center right x [mm]": float,
                "eyeball center right y [mm]": float,
                "eyeball center right z [mm]": float,
                "optical axis left x": float,
                "optical axis left y": float,
                "optical axis left z": float,
                "optical axis right x": float,
                "optical axis right y": float,
                "optical axis right z": float,
                "time [s]": float,
            }
        )


class NeonIMU(NeonStream):
    """
    IMU data.
    """

    def __init__(self, file: Path):
        super().__init__(file)
        self.sampling_freq_nominal = int(110)
        self.data = self.data.astype(
            {
                "timestamp [ns]": "Int64",
                "gyro x [deg/s]": float,
                "gyro y [deg/s]": float,
                "gyro z [deg/s]": float,
                "acceleration x [g]": float,
                "acceleration y [g]": float,
                "acceleration z [g]": float,
                "roll [deg]": float,
                "pitch [deg]": float,
                "yaw [deg]": float,
                "quaternion w": float,
                "quaternion x": float,
                "quaternion y": float,
                "quaternion z": float,
                "time [s]": float,
            }
        )


class NeonEV(NeonData):
    """
    Base for Neon event data (blinks, fixations, saccades, messages).
    """

    def __init__(self, file):
        super().__init__(file)

    def __getitem__(self, index) -> pd.Series:
        """Get an event timeseries by index."""
        return self.data.iloc[index]

    # def _get_attributes(self):
    #     """
    #     Get attributes given self.data DataFrame.
    #     """
    #     # If there is a timestamp [ns] column, rename it to start timestamp [ns]
    #     if "timestamp [ns]" in self.data.columns:
    #         self.data.rename(
    #             columns={"timestamp [ns]": "start timestamp [ns]"}, inplace=True
    #         )
    #         self.data["end timestamp [ns]"] = self.data["start timestamp [ns]"]
    #     self.data.sort_values(by=["start timestamp [ns]"], inplace=True)
    #     self.start_timestamps = self.data["start timestamp [ns]"].to_numpy()
    #     self.start_ts = self.start_timestamps
    #     self.end_timestamps = self.data["end timestamp [ns]"].to_numpy()
    #     self.end_ts = self.end_timestamps


class NeonBlinks(NeonEV):
    def __init__(self, file):
        super().__init__(file)
        self.data = self.data.astype(
            {
                "blink id": "Int32",
                "start timestamp [ns]": "Int64",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
            }
        )


class NeonFixations(NeonEV):
    def __init__(self, file):
        super().__init__(file)
        self.data = self.data.astype(
            {
                "fixation id": "Int32",
                "start timestamp [ns]": "Int64",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
                "fixation x [px]": float,
                "fixation y [px]": float,
                "azimuth [deg]": float,
                "elevation [deg]": float,
            }
        )


class NeonSaccades(NeonEV):
    def __init__(self, file):
        super().__init__(file)
        self.data = self.data.astype(
            {
                "saccade id": "Int32",
                "start timestamp [ns]": "Int64",
                "end timestamp [ns]": "Int64",
                "duration [ms]": "Int64",
                "amplitude [px]": float,
                "amplitude [deg]": float,
                "mean velocity [px/s]": float,
                "peak velocity [px/s]": float,
            }
        )


class NeonEvents(NeonEV):
    def __init__(self, file):
        super().__init__(file)
        self.data = self.data.astype(
            {
                "timestamp [ns]": "Int64",
                "name": str,
                "type": str,
            }
        )


class NeonVideo(cv2.VideoCapture):
    """
    Loaded video file with timestamps.

    Parameters
    ----------
    video_file : :class:`pathlib.Path`
        Path to the video file.
    timestamps_file : :class:`pathlib.Path`
        Path to the timestamps file.

    Attributes
    ----------
    timestamps : np.ndarray
        Timestamps of the video frames.
    ts : np.ndarray
        Alias for timestamps.
    fps : float
        Frames per second of the video.
    width : int
        Width of the video frames in pixels.
    height : int
        Height of the video frames in pixels.
    """

    def __init__(self, video_file, timestamps_file):
        super().__init__(video_file)
        self.timestamps = (
            pd.read_csv(timestamps_file)["timestamp [ns]"].to_numpy().dtype(np.int64)
        )
        self.ts = self.timestamps
        assert len(self.timestamps) == self.get(cv2.CAP_PROP_FRAME_COUNT), (
            f"Number of timestamps ({len(self.timestamps)}) does not match "
            f"number of frames ({self.get(cv2.CAP_PROP_FRAME_COUNT)})"
        )

        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
