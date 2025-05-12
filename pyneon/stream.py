from pathlib import Path
import pandas as pd
import numpy as np
from numbers import Number
from typing import Literal, Optional
import copy

from .tabular import BaseTabular
from .preprocess import interpolate, window_average


class Stream(BaseTabular):
    """
    Base for a continuous data stream (gaze, eye states, IMU).
    Indexed by ``timestamp [ns]``.

    Parameters
    ----------
    data : pandas.DataFrame or pathlib.Path
        DataFrame or path to the CSV file containing the stream data.
        The data must be indexed by ``timestamp [ns]``.

    Attributes
    ----------
    file : pathlib.Path
        Path to the CSV file containing the stream data.
    data : pandas.DataFrame
        DataFrame containing the stream data.
    sampling_freq_nominal : int or None
        Nominal sampling frequency of the stream as specified by Pupil Labs
        (https://pupil-labs.com/products/neon/specs).
    """

    def __init__(
        self, data: pd.DataFrame | Path, sampling_freq_nominal: Optional[int] = None
    ):
        if isinstance(data, Path):
            self.file = data
            data = pd.read_csv(data)
        else:
            self.file = None
        super().__init__(data)
        self.sampling_freq_nominal = sampling_freq_nominal

    def __getitem__(self, index: str) -> pd.Series:
        if index not in self.data.columns:
            raise KeyError(f"Column '{index}' not found in the stream data.")
        return self.data[index]

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps of the stream in nanoseconds."""
        return self.data.index.to_numpy()

    @property
    def ts(self) -> np.ndarray:
        """Alias for timestamps."""
        return self.timestamps

    @property
    def first_ts(self) -> int:
        """First timestamp of the stream."""
        return int(self.ts[0])

    @property
    def last_ts(self) -> int:
        """Last timestamp of the stream."""
        return int(self.ts[-1])

    @property
    def ts_diff(self) -> np.ndarray:
        """Difference between consecutive timestamps."""
        return np.diff(self.ts)

    @property
    def times(self) -> np.ndarray:
        """Timestamps converted to seconds relative to stream start."""
        return (self.ts - self.first_ts) / 1e9

    @property
    def duration(self) -> float:
        """Duration of the stream in seconds."""
        return float(self.times[-1] - self.times[0])

    @property
    def sampling_freq_effective(self) -> float:
        """Effective sampling frequency of the stream."""
        return len(self.data) / self.duration

    @property
    def is_uniformly_sampled(self) -> bool:
        """Whether the stream is uniformly sampled."""
        return np.allclose(self.ts_diff, self.ts_diff[0])

    def time_to_ts(self, time: Number | np.ndarray) -> np.ndarray:
        """Convert relative time(s) in seconds to closest timestamp(s) in nanoseconds."""
        time = np.array([time])
        return np.array([self.ts[np.absolute(self.times - t).argmin()] for t in time])

    def crop(
        self,
        tmin: Optional[Number] = None,
        tmax: Optional[Number] = None,
        by: Literal["timestamp", "time", "row"] = "timestamp",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Crop data to a specific time range based on timestamps,
        relative times since start, or row numbers.

        Parameters
        ----------
        tmin : numbers.Number, optional
            Start timestamp/time/row to crop the data to. If ``None``,
            the minimum timestamp/time/row in the data is used. Defaults to ``None``.
        tmax : numbers.Number, optional
            End timestamp/time/row to crop the data to. If ``None``,
            the maximum timestamp/time/row in the data is used. Defaults to ``None``.
        by : "timestamp" or "time" or "row", optional
            Whether tmin and tmax are UTC timestamps in nanoseconds
            OR relative times in seconds OR row numbers of the stream data.
            Defaults to "timestamp".
        inplace : bool, optional
            Whether to replace the data in the object with the cropped data.
            Defaults to False.

        Returns
        -------
        Stream or None
            Cropped stream if ``inplace=False``, otherwise ``None``.
        """
        if tmin is None and tmax is None:
            raise ValueError("At least one of tmin or tmax must be provided")
        if by == "timestamp":
            t = self.ts
        elif by == "time":
            t = self.times
        else:
            t = np.arange(len(self))
        tmin = tmin if tmin is not None else t.min()
        tmax = tmax if tmax is not None else t.max()
        mask = (t >= tmin) & (t <= tmax)
        if not mask.any():
            raise ValueError("No data found in the specified time range")
        new_data = self.data[mask].copy()
        if inplace:
            self.data = new_data
        else:
            new_stream = copy.copy(self)
            new_stream.data = new_data
            return new_stream

    def restrict(
        self,
        other: "Stream",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Temporally restrict the stream to the timestamps of another stream.
        Equivalent to ``crop(other.first_ts, other.last_ts)``.

        Parameters
        ----------
        other : Stream
            The other stream whose timestamps are used to restrict the data.
        inplace : bool, optional
            Whether to replace the data in the object with the restricted data.

        Returns
        -------
        Stream or None
            Restricted stream if ``inplace=False``, otherwise ``None``.
        """
        new_stream = self.crop(
            other.first_ts, other.last_ts, by="timestamp", inplace=inplace
        )
        if new_stream.data.empty:
            raise ValueError("No data found in the range of the other stream")
        return new_stream

    def interpolate(
        self,
        new_ts: Optional[np.ndarray] = None,
        float_kind: str = "cubic",
        other_kind: str = "nearest",
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Interpolate the stream to a new set of timestamps.

        Parameters
        ----------
        new_ts : numpy.ndarray, optional
            An array of new timestamps (in nanoseconds)
            at which to evaluate the interpolant. If ``None`` (default), new timestamps
            are generated according to the nominal sampling frequency of the stream as
            specified by Pupil Labs: https://pupil-labs.com/products/neon/specs.
        float_kind : str, optional
            Kind of interpolation applied on columns of ``float`` type,
            For details see :class:`scipy.interpolate.interp1d`.
            Defaults to ``"cubic"``.
        other_kind : str, optional
            Kind of interpolation applied on columns of other types,
            For details see :class:`scipy.interpolate.interp1d`.
            Defaults to ``"nearest"``.
        inplace : bool, optional
            Whether to replace the data in the object with the interpolated data.
            Defaults to False.

        Returns
        -------
        Stream or None
            Interpolated stream if ``inplace=False``, otherwise ``None``.
        """
        # If new_ts is not provided, generate a evenly spaced array of timestamps
        if new_ts is None:
            if self.sampling_freq_nominal is None:
                raise ValueError(
                    "The nominal sampling frequency of the stream is not specified, "
                    "please specify new_ts manually."
                )
            step_size = int(1e9 / self.sampling_freq_nominal)
            new_ts = np.arange(self.first_ts, self.last_ts, step_size, dtype=np.int64)
            assert new_ts[0] == self.first_ts
            assert np.allclose(np.diff(new_ts), step_size)

        new_data = interpolate(new_ts, self.data, float_kind, other_kind)
        if inplace:
            self.data = new_data
        else:
            new_instance = copy.copy(self)
            new_instance.data = new_data
            return new_instance

    def window_average(
        self,
        new_ts: np.ndarray,
        window_size: Optional[int] = None,
        inplace: bool = False,
    ) -> Optional["Stream"]:
        """
        Take the average over a time window to obtain smoothed data at new timestamps.

        Parameters
        ----------
        new_ts : numpy.ndarray
            An array of new timestamps (in nanoseconds)
            at which to compute the windowed averages.
            The median interval between these new timestamps must be larger than
            the median interval between the original data timestamps, i.e.,
            ``np.median(np.diff(new_ts)) > np.median(np.diff(data.index))``.
            In other words, only downsampling is supported.
        window_size : int, optional
            The size of the time window (in nanoseconds)
            over which to compute the average around each new timestamp.
            If ``None`` (default), the window size is set to the median interval
            between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
            The window size must be larger than the median interval between the original data timestamps,
            i.e., ``window_size > np.median(np.diff(data.index))``.
        inplace : bool, optional
            Whether to replace the data in the object with the window averaged data.
            Defaults to False.

        Returns
        -------
        Stream or None
            Stream with window average applied on data if ``inplace=False``, otherwise ``None``.
        """
        new_data = window_average(new_ts, self.data, window_size)
        if inplace:
            self.data = new_data
        else:
            new_instance = copy.copy(self)
            new_instance.data = new_data
            return new_instance
