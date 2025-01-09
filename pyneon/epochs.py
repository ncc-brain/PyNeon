import pandas as pd
import numpy as np

from numbers import Number
from typing import TYPE_CHECKING, Literal
import warnings

from .events import (
    NeonEV,
    NeonBlinks,
    NeonFixations,
    NeonSaccades,
    NeonEvents,
    CustomEvents,
)

if TYPE_CHECKING:
    from .tabular import NeonTabular


class Epochs:
    """
    Class to create and manage epochs in the data streams.

    Parameters
    ----------
    source : NeonTabular
        Data stream to create epochs from.
    times_df : pd.DataFrame, optional
        DataFrame containing epoch information with the following columns:
        - 't_ref': Reference time of the epoch, in nanoseconds.
        - 't_before': Time before the reference time to start the epoch, in nanoseconds.
        - 't_after': Time after the reference time to end the epoch, in nanoseconds.
        - 'description': Description or label associated with the epoch.
        If provided, `t_ref`, `t_before`, `t_after`, `description`, `global_t_ref`, and `time_unit` are ignored.
    t_ref : np.ndarray or list, optional
        Array or list of reference times for the epochs. Units specified by `time_unit`.
    t_before : float, np.ndarray, or list, optional
        Time before the reference time to start the epoch, in **seconds**.
    t_after : float, np.ndarray, or list, optional
        Time after the reference time to end the epoch, in **seconds**.
    description : str, np.ndarray, or list, optional
        Description or label associated with the epoch.
    global_t_ref : int or float, optional
        Global reference time to be added to each reference time in `t_ref`. Units specified by `time_unit`. Default is 0.
    time_unit : str, optional
        Unit of time for the reference times and `global_t_ref` ('ns' for nanoseconds or 's' for seconds). Default is 'ns'.

    Notes
    -----
    - If `times_df` is provided, it is used to create epochs, and the other time-related parameters are ignored.
    - If `times_df` is not provided, `t_ref`, `t_before`, `t_after`, and `description` must be provided.
    - The `t_before` and `t_after` parameters are always expected in **seconds** and will be converted to nanoseconds internally.
    """

    def __init__(
        self,
        source: "NeonTabular",
        times_df: pd.DataFrame | None = None,
        t_ref: np.ndarray | None = None,
        t_before: np.ndarray | Number | None = None,
        t_after: np.ndarray | Number | None = None,
        description: np.ndarray | str | None = None,
        global_t_ref: int = 0,
        time_unit: Literal["s", "ms", "us", "ns"] = "ns",
    ):
        if times_df is not None:
            if times_df.isnull().values.any():
                raise ValueError("times_df should not have any empty values")
            # Set columns to appropriate data types (check if columns are present along the way)
            times_df = times_df.astype(
                {
                    "t_ref": "int64",
                    "t_before": "int64",
                    "t_after": "int64",
                    "description": "str",
                }
            )
        else:
            # Ensure the input arrays are not None
            if any(x is None for x in [t_ref, t_before, t_after, description]):
                raise ValueError(
                    "t_ref, t_before, t_after, and description must be provided if times_df is None"
                )
            times_df = construct_times_df(
                t_ref,
                t_before,
                t_after,
                description,
                global_t_ref,
                time_unit,
            )

        # Create epochs
        self.epochs, self.data = _create_epochs(source, times_df)

        # # Check epoch lengths
        # data_len = self.epochs["epoch data"].apply(lambda x: x.shape[0])
        # self.min_len = data_len.min()
        # self.max_len = data_len.max()
        # self.equal_times = data_len.nunique() == 1

        # # Check if t_ref differences are equal
        # t_ref_diff = self.epochs["t_ref"].diff().dropna().unique()
        # self.equal_dist = len(t_ref_diff) == 1

        # # Check if t_before and t_after are the same across epochs
        # self.equal_length = (
        #     self.epochs["t_before"].nunique() == 1
        #     and self.epochs["t_after"].nunique() == 1
        # )
        # if self.equal_length:
        #     self.window_length = (
        #         self.epochs["t_before"].iloc[0] + self.epochs["t_after"].iloc[0]
        #     )

    def to_numpy(self, sampling_rate=100, columns=None):
        """
        Converts epochs into a NumPy array with dimensions (n_epochs, n_times, n_channels).
        Resamples epochs to a fixed sampling rate.

        Parameters
        ----------
        sampling_rate : int
            The sampling rate to resample the data to, in **Hz** (samples per second).
        columns : list of str, optional
            List of column names to extract from the DataFrame. If None, all columns except 't_rel' are used.

        Returns
        -------
        epochs_np : np.ndarray
            NumPy array of shape (n_epochs, n_times, n_channels).
        info : dict
            A dictionary containing:
            - 'column_ids': List of provided column names.
            - 't_rel': The common time grid, in nanoseconds.
            - 'nan_status': String indicating whether NaN values were found in the data.

        Notes
        -----
        - The time grid (`t_rel`) is in nanoseconds.
        - If `NaN` values are present after interpolation, they are noted in `nan_status`.
        """
        # Ensure there are epochs to process
        if len(self.epochs) == 0:
            raise ValueError("No epochs with data to convert to NumPy array.")

        # Remove epochs with empty data
        self.epochs = self.epochs[self.epochs["epoch data"].apply(len) > 0].reset_index(
            drop=True
        )
        n_epochs = len(self.epochs)

        # Define the common time grid
        t_before = self.epochs["t_before"].iloc[0]
        t_after = self.epochs["t_after"].iloc[0]
        total_duration = t_after + t_before
        n_times = int(total_duration / 1e9 * sampling_rate) + 1
        common_times = np.linspace(-t_before, t_after, n_times)

        # Select the relevant data columns
        if columns is None:
            # If no columns are provided, use all columns except 't_rel'
            data_columns = self.epochs.iloc[0]["epoch data"].columns.drop("t_rel")
        else:
            # Use the explicitly provided columns
            data_columns = [
                col
                for col in columns
                if col in self.epochs.iloc[0]["epoch data"].columns
            ]

        if len(data_columns) == 0:
            raise ValueError("None of the provided columns exist in the epoch data.")

        n_channels = len(data_columns)

        # Initialize the NumPy array
        # MNE convention: (n_epochs, n_channels, n_times)
        epochs_np = np.full((n_epochs, n_times, n_channels), np.nan)

        # Interpolate each epoch onto the common time grid
        for i, (_, epoch) in enumerate(self.epochs.iterrows()):
            epoch_data = epoch["epoch data"].copy()
            t_rel = epoch_data["t_rel"].values
            for idx, col in enumerate(data_columns):
                y = epoch_data[col].values
                # Interpolate using numpy.interp
                interp_values = np.interp(
                    common_times, t_rel, y, left=np.nan, right=np.nan
                )
                epochs_np[i, :, idx] = interp_values

        # check if there are any NaN values in the data
        nan_flag = np.isnan(epochs_np).any()
        if nan_flag:
            nan_text = "NaN values were found in the data."
        else:
            nan_text = "No NaN values were found in the data."

        # Return an object holding the column ids, times, and data
        info = {
            "column_ids": data_columns,
            "t_rel": common_times,
            "nan_status": nan_text,
        }
        print(nan_text)

        return epochs_np, info

    def __len__(self):
        return len(self.epochs)


def _create_epochs(
    source: "NeonTabular", times_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create epochs DataFrame and annotate the data with epoch information.
    """
    data = source.data.copy()
    data["epoch id"] = pd.Series(dtype="Int32")
    data["epoch time"] = pd.Series(dtype="int64")
    data["epoch description"] = pd.Series(dtype="str")
    ts = source.ts

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

        data.loc[mask, "epoch id"] = i
        data.loc[mask, "epoch description"] = str(description_i)
        data.loc[mask, "epoch time"] = (
            data.loc[mask].index.to_numpy() - t_ref_i
        ).astype("int64")

        local_data = data.loc[mask].copy()
        local_data.drop(columns=["epoch id", "epoch description"], inplace=True)
        epochs.at[i, "data"] = local_data

    return epochs, data


def events_to_times_df(
    event: "NeonEV",
    t_before: Number,
    t_after: Number,
    time_unit: Literal["s", "ms", "us", "ns"] = "s",
    type_name: str = "all",
) -> pd.DataFrame:
    """
    Construct a times_df DataFrame suitable for creating epochs from event data.

    Parameters
    ----------
    event : NeonEV
        NeonEV instance containing the event times.
    t_before : Number
        Time before the event start time to start the epoch. Units specified by `time_unit`.
    t_after : Number
        Time after the event start time to end the epoch. Units specified by `time_unit`.
    time_unit : str, optional
        Unit of time for the reference times, can be 's' (seconds), 'ms' (milliseconds),
        'us' (microseconds), or 'ns' (nanoseconds). Defaults to 's'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing epoch information with the following columns:

            ``t_ref``: Reference time of the epoch, in nanoseconds.

            ``t_before``: Time before the reference time to start the epoch, in nanoseconds.

            ``t_after``: Time after the reference time to end the epoch, in nanoseconds.

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
        if "type" not in event.data.columns:
            raise ValueError("Custom event data must have a 'type' column.")
        if type_name == "all":
            description = event.data["type"].to_numpy()
        else:
            mask = event.data["type"] == type_name
            t_ref = event.data.index.to_numpy()[mask]
            description = type_name

    times_df = construct_times_df(
        t_ref,
        t_before,
        t_after,
        description,
        0,
        time_unit,
    )
    times_df["t_ref"] = t_ref
    return times_df


def construct_times_df(
    t_ref: np.ndarray,
    t_before: np.ndarray | Number,
    t_after: np.ndarray | Number,
    description: np.ndarray | str,
    global_t_ref: int = 0,
    time_unit: Literal["s", "ms", "us", "ns"] = "ns",
) -> pd.DataFrame:
    """
    Handles the construction of the times_df DataFrame for creating epochs. It populates
    single values for `t_before`, `t_after`, and `description` to match the length of `t_ref`.
    and converts all times to UTC timestamps in nanoseconds.

    Parameters
    ----------
    t_ref : array of Number, shape (n_epoch,)
        Array or list of reference times for the epochs. Units specified by `time_unit`.
    t_before : array of Number, shape (n_epoch,) or Number
        Time before the reference time to start the epoch. Units specified by `time_unit`.
    t_after : array of Number, shape (n_epoch,) or Number
        Time after the reference time to end the epoch. Units specified by `time_unit`.
    description : array of str, shape (n_epoch,) or str
        Description or label associated with the epoch.
    global_t_ref : int
        Global reference time to be added to each reference time in `t_ref`.
        Should represent UTC timestamp in nanoseconds. Defaults to 0.
    time_unit : str
        Unit of time for the reference times and `global_t_ref` ('ns' for nanoseconds or 's' for seconds).
        Defaults to 'ns'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing epoch information with the following columns:

            ``t_ref``: Reference time of the epoch, in nanoseconds.

            ``t_before``: Time before the reference time to start the epoch, in nanoseconds.

            ``t_after``: Time after the reference time to end the epoch, in nanoseconds.

            ``description``: Description or label associated with the epoch.
    """
    n_epoch = len(t_ref)

    # Set conversion factor based on time unit
    match time_unit:
        case "s":
            factor = 1e9
        case "ms":
            factor = 1e6
        case "us":
            factor = 1e3
        case "ns":
            factor = 1

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
            raise ValueError(f"{name} must be a single number or a numpy array")

    # Construct the event times DataFrame
    # Do rounding as they should be timestamps already
    times_df = pd.DataFrame(
        {
            "t_ref": t_ref * factor + global_t_ref,
            "t_before": t_before * factor,
            "t_after": t_after * factor,
            "description": description,
        }
    )

    return times_df
