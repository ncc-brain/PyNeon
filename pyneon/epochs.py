import pandas as pd
import numpy as np

from numbers import Number
from typing import TYPE_CHECKING, Literal
import warnings

from .stream import NeonStream
from .events import (
    NeonEV,
    NeonBlinks,
    NeonFixations,
    NeonSaccades,
    NeonEvents,
    CustomEvents,
)


def _check_overlap(times_df: pd.DataFrame) -> bool:
    """
    Emits warnings if any adjacent epochs overlap in time.
    """
    times_df = times_df.sort_values("t_ref")
    overlap = False
    overlap_epochs = []
    for i in range(1, times_df.shape[0]):
        # Check if the current epoch overlaps with the previous epoch
        if (
            times_df["t_ref"].iloc[i] - times_df["t_before"].iloc[i]
            < times_df["t_ref"].iloc[i - 1] + times_df["t_after"].iloc[i - 1]
        ):
            overlap_epochs.append((i - 1, i))
            overlap = True
    if overlap:
        warnings.warn(
            f"The following epochs overlap in time:\n{overlap_epochs}", RuntimeWarning
        )
    return overlap


class Epochs:
    """
    Class to create and manage epochs in the data streams.

    Parameters
    ----------
    source : NeonStream or NeonEV
        Data to create epochs from.
    times_df : pandas.DataFrame, shape (n_epochs, 4), optional
        DataFrame containing epoch information with the following columns:

            ``t_ref``: Reference time of the epoch, in nanoseconds.\n
            ``t_before``: Time before the reference time to start the epoch, in nanoseconds.\n
            ``t_after``: Time after the reference time to end the epoch, in nanoseconds.\n
            ``description``: Description or label associated with the epoch.

        Must not have empty values. If provided, the rest of the parameters are ignored.
    t_ref : numpy.ndarray, optional
        Array of reference times for the epochs. Units specified by ``t_ref_unit``.
    t_before : numpy.ndarray or Number, optional
        Time before the reference time to start the epoch. Could be an array of
        equal length as ``t_ref`` or a single number (to be repeated for all epochs).
        Units specified by ``t_other_unit``.
    t_after : numpy.ndarray or Number, optional
        Time after the reference time to end the epoch. Could be an array of
        equal length as ``t_ref`` or a single number (to be repeated for all epochs).
        Units specified by ``t_other_unit``.
    description : numpy.ndarray or str, optional
        Description or label associated with the epochs. Could be an array of
        equal length as ``t_ref`` or a single string (to be repeated for all epochs).
    global_t_ref : int, optional
        Global reference time (in nanoseconds) to be added to `t_ref`.
        Unit is nanosecond. Defaults to 0. This is useful when the reference times
        are relative to a global start time
        (for instance :attr:`pyneon.stream.NeonStream.first_ts`).
    t_ref_unit : str, optional
        Unit of time for the reference times. Default is 'ns'.
    t_other_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``. Default is 's'.

    Notes
    -----
    An epoch spans the temporal range of ``t_ref - t_before`` to ``t_ref + t_after`` as shown below:

    ::

                        t_ref[0]                            t_ref[1]
            <--t_before[0]--|--t_after[0]-->      <-t_before[1]-|-t_after[1]->
        ├--------------------------------------------------------------------------------┤

    If ``times_df`` is provided, it is used to create epochs, and the other
    time-related parameters are ignored. Otherwise, ``t_ref``, ``t_before``,
    ``t_after``, and ``description`` are required.

    Attributes
    ----------
    epochs : pandas.DataFrame
        DataFrame containing epoch information with the following columns:

            ``t_ref`` (int64): Reference time of the epoch, in nanoseconds.\n
            ``t_before`` (int64): Time before the reference time to start the epoch, in nanoseconds.\n
            ``t_after`` (int64): Time after the reference time to end the epoch, in nanoseconds.\n
            ``description`` (str): Description or label associated with the epoch.\n
            ``data`` (object): DataFrame containing the data for each epoch.
    data : pandas.DataFrame
        Annotated data with epoch information. In addition to the original data columns,
        the following columns are added:

            ``epoch index`` (Int32): ID of the epoch the data belongs to.\n
            ``epoch time`` (Int64): Time relative to the epoch reference time, in nanoseconds.\n
            ``epoch description`` (str): Description or label associated with the epoch.

        If epochs overlap, data annotations are always overwritten by the latest epoch.
    """

    def __init__(
        self,
        source: NeonStream | NeonEV,
        times_df: pd.DataFrame | None = None,
        t_ref: np.ndarray | None = None,
        t_before: np.ndarray | Number | None = None,
        t_after: np.ndarray | Number | None = None,
        description: np.ndarray | str | None = None,
        t_ref_unit: Literal["s", "ms", "us", "ns"] = "ns",
        t_other_unit: Literal["s", "ms", "us", "ns"] = "s",
        global_t_ref: int = 0,
    ):
        if times_df is not None:
            if times_df.isnull().values.any():
                raise ValueError("times_df should not have any empty values")
        else:
            # Ensure the input arrays are not None
            if any(x is None for x in [t_ref, t_before, t_after, description]):
                raise ValueError(
                    "t_ref, t_before, t_after, and description must be provided if times_df is None"
                )
            times_df = _construct_times_df(
                t_ref,
                t_before,
                t_after,
                description,
                t_ref_unit,
                t_other_unit,
                global_t_ref,
            )

        # Sort by t_ref
        assert times_df.shape[0] > 0, "times_df must have at least one row"
        times_df = times_df.sort_values("t_ref").reset_index(drop=True)
        # Set columns to appropriate data types (check if columns are present along the way)
        times_df = times_df.astype(
            {
                "t_ref": "int64",
                "t_before": "int64",
                "t_after": "int64",
                "description": "str",
            }
        )

        if isinstance(source, NeonStream):
            self.source_type = "stream"
            self.is_uniformly_sampled = source.is_uniformly_sampled
            self.sf = source.sampling_freq_effective
        elif isinstance(source, NeonEV):
            self.source_type = "event"
            self.is_uniformly_sampled = None
            self.sf = None

        # Create epochs
        self.epochs, self.data = _create_epochs(source, times_df)

    def __len__(self):
        return self.epochs.shape[0]

    @property
    def t_ref(self) -> np.ndarray:
        """The reference time for each epoch in UTC nanoseconds."""
        return self.epochs["t_ref"].to_numpy()

    @property
    def t_before(self) -> np.ndarray:
        """The time before the reference time for each epoch in nanoseconds."""
        return self.epochs["t_before"].to_numpy()

    @property
    def t_after(self) -> np.ndarray:
        """The time after the reference time for each epoch in nanoseconds."""
        return self.epochs["t_after"].to_numpy()

    @property
    def description(self) -> np.ndarray:
        """The description or label for each epoch."""
        return self.epochs["description"].to_numpy()

    @property
    def columns(self) -> pd.Index:
        return self.data.columns[:-3]

    @property
    def dtypes(self) -> pd.Series:
        """The data types of the epoched data."""
        return self.data.dtypes[:-3]

    @property
    def is_equal_length(self) -> bool:
        """Whether all epochs have the same length."""
        return np.allclose(self.t_before, self.t_before[0]) and np.allclose(
            self.t_after, self.t_after[0]
        )

    @property
    def has_overlap(self) -> bool:
        """Whether any adjacent epochs overlap."""
        return _check_overlap(self.epochs)

    def to_numpy(
        self,
        column_names: str | list[str] = "all",
    ) -> tuple[np.ndarray, dict]:
        """
        Converts epochs into a 3D array with dimensions (n_epochs, n_channels, n_times).
        Acts similarly as :meth:`mne.Epochs.get_data`.
        Requires the epoch to be created from a uniformly-sampled :class:`pyneon.stream.NeonStream`.

        Parameters
        ----------
        columns : str or list of str, optional
            Column names to include in the NumPy array. If 'all', all columns are included.
            Only columns that can be converted to int or float can be included.

        Returns
        -------
        numpy_epochs : numpy.ndarray
            NumPy array of shape (n_epochs, n_channels, n_times).

        info : dict
            A dictionary containing:

                ``"column_ids"``: List of provided column names.\n
                ``"t_rel"``: The common time grid, in nanoseconds.\n
                ``"nan_flag"``: String indicating whether NaN values were found in the data.

        Notes
        -----
        - The time grid (`t_rel`) is in nanoseconds.
        - If `NaN` values are present after interpolation, they are noted in `nan_flag`.
        """
        if self.source_type != "stream" or self.is_uniformly_sampled is False:
            raise ValueError(
                "The source must be a uniformly-sampled NeonStream to convert to NumPy array."
            )
        if not self.is_equal_length:
            raise ValueError("Epochs must have equal length to convert to NumPy array.")

        t_before = self.t_before[0]
        t_after = self.t_after[0]

        times = np.linspace(
            -t_before, t_after, int((t_before + t_after) * self.sf * 1e-9) + 1
        )
        n_times = len(times)

        if column_names == "all":
            columns = self.columns.to_list()
        else:
            columns = [column_names] if isinstance(column_names, str) else column_names
            if not all(col in self.columns for col in columns):
                raise ValueError(f"Column '{col}'' doesn't exist in the epoch data.")

        n_columns = len(columns)

        # Initialize the NumPy array
        # MNE convention: (n_epochs, n_channels, n_times)
        epochs_np = np.full((len(self), n_columns, n_times - 2), np.nan)

        # Interpolate each epoch onto the common time grid
        for i, epoch in self.epochs.iterrows():
            epoch_data = epoch["data"].copy()
            epoch_time = epoch_data["epoch time"].to_numpy()
            for j, col in enumerate(columns):
                y = epoch_data[col].to_numpy()
                interp_values = np.interp(
                    times, epoch_time, y, left=np.nan, right=np.nan
                )
                interp_values = interp_values[1:-1]  # Exclude the first and last values
                epochs_np[i, j, :] = interp_values

        # check if there are any NaN values in the data
        nan_flag = np.isnan(epochs_np).any()
        if nan_flag:
            warnings.warn("NaN values were found in the data.", RuntimeWarning)

        # Return an object holding the column ids, times, and data
        info = {
            "column_ids": columns,
            "epoch_times": times[1:-1] * 1e-9,  # Convert to seconds
            "nan_flag": nan_flag,
        }

        return epochs_np, info


def _create_epochs(
    source: NeonStream | NeonEV, times_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create epochs DataFrame and annotate the data with epoch information.
    """
    _check_overlap(times_df)

    data = source.data.copy()
    data["epoch index"] = pd.Series(dtype="Int32")
    data["epoch time"] = pd.Series(dtype="Int64")
    data["epoch description"] = pd.Series(dtype="str")

    # check for source type
    if isinstance(source, NeonStream):
        ts = source.ts
    elif isinstance(source, NeonEV):
        ts = source.start_ts
    else:
        raise ValueError("Source must be a NeonStream or NeonEV.")

    epochs = times_df.copy().reset_index(drop=True)
    epochs["data"] = pd.Series(dtype="object")

    # Iterate over each event time to create epochs
    for i, row in times_df.iterrows():
        t_ref_i, t_before_i, t_after_i, description_i = row[
            ["t_ref", "t_before", "t_after", "description"]
        ].to_list()

        start_time = t_ref_i - t_before_i
        end_time = t_ref_i + t_after_i
        mask = np.logical_and(ts >= start_time, ts <= end_time)

        if not mask.any():
            warnings.warn(f"No data found for epoch {i}.", RuntimeWarning)
            epochs.at[i, "epoch data"] = pd.DataFrame()
            continue

        data.loc[mask, "epoch index"] = i
        data.loc[mask, "epoch description"] = str(description_i)
        data.loc[mask, "epoch time"] = (
            data.loc[mask].index.to_numpy() - t_ref_i
        ).astype("int64")

        local_data = data.loc[mask].copy()
        local_data.drop(columns=["epoch index", "epoch description"], inplace=True)
        epochs.at[i, "data"] = local_data

    return epochs, data


def events_to_times_df(
    event: "NeonEV",
    t_before: Number,
    t_after: Number,
    t_unit: Literal["s", "ms", "us", "ns"] = "s",
    type_name: str = "all",
) -> pd.DataFrame:
    """
    Construct a times_df DataFrame suitable for creating epochs from event data.

    Parameters
    ----------
    event : NeonEV
        NeonEV instance containing the event times.
    t_before : numbers.Number
        Time before the event start time to start the epoch. Units specified by `t_unit`.
    t_after : numbers.Number
        Time after the event start time to end the epoch. Units specified by `t_unit`.
    t_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``. Can be 's' (seconds), 'ms' (milliseconds),
        'us' (microseconds), or 'ns' (nanoseconds). Defaults to 's'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing epoch information with the following columns:

            ``t_ref``: Reference time of the epoch, in nanoseconds.\n
            ``t_before``: Time before the reference time to start the epoch, in nanoseconds.\n
            ``t_after``: Time after the reference time to end the epoch, in nanoseconds.\n
            ``description``: Description or label associated with the epoch.
    """

    if isinstance(event, NeonBlinks):
        description = "blink"
    elif isinstance(event, NeonFixations):
        description = "fixation"
    elif isinstance(event, NeonSaccades):
        description = "saccade"
    elif isinstance(event, NeonEvents):
        description = "event"
    t_ref = event.start_ts

    if isinstance(event, CustomEvents):
        if "name" not in event.data.columns:
            raise ValueError("Custom event data must have a 'name' column.")
        if type_name == "all":
            description = event.data["name"].to_numpy()
            t_ref = event.data.index.to_numpy()
        else:
            mask = event.data["name"] == type_name
            t_ref = event.data.index.to_numpy()[mask]
            description = event.data["name"].to_numpy()[mask]

    times_df = _construct_times_df(
        t_ref,
        t_before,
        t_after,
        description,
        "ns",
        t_unit,
    )
    return times_df


def _construct_times_df(
    t_ref: np.ndarray,
    t_before: np.ndarray | Number,
    t_after: np.ndarray | Number,
    description: np.ndarray | str,
    t_ref_unit: Literal["s", "ms", "us", "ns"] = "ns",
    t_other_unit: Literal["s", "ms", "us", "ns"] = "s",
    global_t_ref: int = 0,
) -> pd.DataFrame:
    """
    Handles the construction of the times_df DataFrame for creating epochs. It populates
    single values for `t_before`, `t_after`, and `description` to match the length of `t_ref`.
    and converts all times to UTC timestamps in nanoseconds.
    """

    if n_epoch := len(t_ref) == 0:
        raise ValueError("t_ref must not be empty")
    else:
        n_epoch = len(t_ref)

    time_factors = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1}

    # Check each of the parameters (t_before, t_after, description) for correct length and type
    for x, name in zip(
        [t_before, t_after, description], ["t_before", "t_after", "description"]
    ):
        if isinstance(x, np.ndarray):
            # Ensure it's the same length as t_ref
            if len(x) != n_epoch:
                raise ValueError(f"{name} must have the same length as t_ref")
        elif isinstance(x, (Number, str)):
            x = np.repeat(x, n_epoch)
        else:
            raise ValueError(f"{name} must be a single value or a numpy array")

    # Construct the event times DataFrame
    # Do rounding as they should be timestamps already
    times_df = pd.DataFrame(
        {
            "t_ref": t_ref * time_factors[t_ref_unit] + global_t_ref,
            "t_before": t_before * time_factors[t_other_unit],
            "t_after": t_after * time_factors[t_other_unit],
            "description": description,
        }
    )

    return times_df
