from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union
from .preprocess import resample, concat_channels


class NeonData:
    """Holder for all formats Neon tabular data."""

    def __init__(self, file: Path):
        self.file = file
        if isinstance(file, Path) and file.suffix == ".csv":
            data = pd.read_csv(file)
        else:  # TODO: Implement reading native data formats
            pass
        # Assert that all section id are the same
        if data["section id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple section IDs")
        if data["recording id"].nunique() > 1:
            raise ValueError(f"{file.name} contains multiple recording IDs")
        self.data = data.drop(columns=["section id", "recording id"])

    def __len__(self) -> int:
        return self.data.shape[0]


class NeonSignal(NeonData):
    """Holder for continuous signals."""

    def __init__(self, file):
        super().__init__(file)
        # Enforce that the data is sorted by timestamp
        self.data.sort_values(by=["timestamp [ns]"], inplace=True)
        self.data["time [s]"] = self.to_seconds()

    def to_seconds(self) -> np.ndarray:
        """Converts the timestamps to seconds."""
        return (
            self.data["timestamp [ns]"] - self.data["timestamp [ns]"][0]
        ).to_numpy() / 1e9

    @property
    def timestamps(self) -> np.ndarray:
        """Returns the timestamps of the stream in nanoseconds."""
        return self.data["timestamp [ns]"].to_numpy()

    @property
    def ts(self) -> np.ndarray:
        """An alias for timestamps."""
        return self.timestamps

    @property
    def first_ts(self) -> int:
        """Returns the first timestamp of the stream in nanoseconds."""
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Returns the last timestamp of the stream in nanoseconds."""
        return int(self.ts[-1])

    @property
    def times(self) -> np.ndarray:
        """Returns the timestamps of the stream in seconds."""
        return self.data["time [s]"].to_numpy()

    @property
    def duration(self) -> float:
        """Returns the duration of the stream in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def sampling_freq_effective(self) -> float:
        """Returns the effective sampling frequency of the stream in Hz."""
        return self.data.shape[0] / self.duration()

    def resample(
        self,
        new_ts: Union[None, np.ndarray] = None,
        float_kind="linear",
        other_kind="nearest",
    ) -> pd.DataFrame:
        """Resample the signal to a new set of timestamps.

        If no new timestamps are provided, the signal is resampled to the nominal sampling rate.
        """
        # If new_ts is not provided, generate a evenly spaced array of timestamps
        if new_ts is None:
            step_size = int(1e9 / self.sampling_rate_nominal)
            new_ts = np.arange(self.first_ts, self.last_ts, step_size, dtype=np.int64)
            assert new_ts[0] == self.first_ts
            assert np.all(np.diff(new_ts) == step_size)
        return resample(new_ts, self.data, float_kind, other_kind)


class NeonGaze(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 200
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


class NeonIMU(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 120
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


class NeonEyeStates(NeonSignal):
    def __init__(self, file):
        super().__init__(file)
        self.sampling_rate_nominal = 200
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


class NeonEvents(NeonData):
    def __init__(self, file):
        super().__init__(file)


class NeonBlinks(NeonEvents):
    def __init__(self, file):
        super().__init__(file)


class NeonFixations(NeonData):
    def __init__(self, file):
        super().__init__(file)


class NeonLabels(NeonData):
    def __init__(self, file):
        super().__init__(file)
