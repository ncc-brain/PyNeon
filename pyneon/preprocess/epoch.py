import pandas as pd
import numpy as np

from numbers import Number
from typing import TYPE_CHECKING, Union


import numpy as np
import pandas as pd


class Epoch:
    """
    Class to create and manage epochs in the data streams.

    Parameters
    ----------
    data : pd.DataFrame
        Data stream to create epochs from. Must contain a 'timestamp [ns]' or 'start timestamp [ns]' column.
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
        data: pd.DataFrame,
        times_df: Union[pd.DataFrame, None] = None,
        t_ref: Union[np.ndarray, None] = None,
        t_before: Union[np.ndarray, Number, None] = None,
        t_after: Union[np.ndarray, Number, None] = None,
        description: Union[np.ndarray, None] = None,
        global_t_ref: Union[int, float] = 0,
        time_unit: str = "ns",
    ):
        self.data = data
        self.times = times_df

        # Check if data is uniformly sampled
        ts_diff = data["timestamp [ns]"].diff().dropna().unique()
        self.uniform_data = len(ts_diff) == 1

        # Create epochs
        self.epochs, self.data = create_epoch(
            data,
            times_df,
            t_ref,
            t_before,
            t_after,
            description,
            global_t_ref,
            time_unit,
        )

        # Check epoch lengths
        data_len = self.epochs["epoch data"].apply(lambda x: x.shape[0])
        self.min_len = data_len.min()
        self.max_len = data_len.max()
        self.equal_times = data_len.nunique() == 1

        # Check if t_ref differences are equal
        t_ref_diff = self.epochs["t_ref"].diff().dropna().unique()
        self.equal_dist = len(t_ref_diff) == 1

        # Check if t_before and t_after are the same across epochs
        self.equal_length = (
            self.epochs["t_before"].nunique() == 1
            and self.epochs["t_after"].nunique() == 1
        )
        if self.equal_length:
            self.window_length = (
                self.epochs["t_before"].iloc[0] + self.epochs["t_after"].iloc[0]
            )

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


def create_epoch(
    data: pd.DataFrame,
    times_df: Union[pd.DataFrame, None] = None,
    t_refs: Union[list, np.ndarray, None] = None,
    t_before: Union[np.ndarray, float, None] = None,
    t_after: Union[np.ndarray, float, None] = None,
    description: Union[np.ndarray, str, None] = None,
    global_t_ref: Union[int, float] = 0,
    time_unit: str = "ns",
):
    """
    Create epochs in the data streams based on the input epochs DataFrame or provided times.

    Parameters
    ----------
    data : pd.DataFrame
        Data stream to create epochs from. Must contain a 'timestamp [ns]' or 'start timestamp [ns]' column.
    times_df : pd.DataFrame, optional
        DataFrame containing epoch information with the following columns:
        - 't_ref': Reference time of the epoch, in nanoseconds.
        - 't_before': Time before the reference time to start the epoch, in nanoseconds.
        - 't_after': Time after the reference time to end the epoch, in nanoseconds.
        - 'description': Description or label associated with the epoch.
        If provided, other time-related parameters are ignored.
    t_refs : list or np.ndarray, optional
        List or array of reference times for the epochs. Units specified by `time_unit`.
    t_before : float, np.ndarray, or list, optional
        Time before the reference time to start the epoch, in **seconds**.
    t_after : float, np.ndarray, or list, optional
        Time after the reference time to end the epoch, in **seconds**.
    description : str, np.ndarray, or list, optional
        Description or label associated with the epoch.
    global_t_ref : int or float, optional
        Global reference time to be added to each reference time in `t_refs`. Units specified by `time_unit`. Default is 0.
    time_unit : str, optional
        Unit of time for the reference times and `global_t_ref` ('ns' for nanoseconds or 's' for seconds). Default is 'ns'.

    Returns
    -------
    epochs : pd.DataFrame
        DataFrame where each row corresponds to an epoch, containing the data belonging to the epoch as a nested DataFrame.
        Columns include:
        - 'epoch id': Unique identifier for the epoch.
        - 't_ref': Reference time of the epoch, in nanoseconds.
        - 't_before': Time before the reference time to start the epoch, in nanoseconds.
        - 't_after': Time after the reference time to end the epoch, in nanoseconds.
        - 'description': Description or label associated with the epoch.
        - 'epoch data': DataFrame containing the data within the epoch.
    annotated_data : pd.DataFrame
        Original data with added columns:
        - 'epoch id': Identifier of the epoch to which the data point belongs.
        - 't_rel': Time relative to the epoch reference time, in nanoseconds.
        - 'description': Description or label associated with the epoch.

    Notes
    -----
    - If `times_df` is provided, it is used to create epochs, and other time-related parameters are ignored.
    - If `times_df` is not provided, `t_refs`, `t_before`, `t_after`, and `description` must be provided.
    - The `t_before` and `t_after` parameters are always expected in **seconds** and will be converted to nanoseconds internally.
    """

    # Determine the timestamp column name
    if "timestamp [ns]" in data.columns:
        ts_name = "timestamp [ns]"
    elif "start timestamp [ns]" in data.columns:
        ts_name = "start timestamp [ns]"
    else:
        raise ValueError(
            "Data must contain a 'timestamp [ns]' or 'start timestamp [ns]' column."
        )

    # Generate event_times DataFrame
    if times_df is not None:
        # Ensure the DataFrame has the required columns
        if not all(
            col in times_df.columns
            for col in ["t_ref", "t_before", "t_after", "description"]
        ):
            raise ValueError(
                "times_df must contain 't_ref', 't_before', 't_after', and 'description' columns"
            )
        event_times = times_df
    else:
        # Ensure the input arrays are not None
        if any(x is None for x in [t_refs, t_before, t_after, description]):
            raise ValueError(
                "t_refs, t_before, t_after, and description must be provided if times_df is None"
            )
        # Use construct_event_times to create the event_times DataFrame
        event_times = construct_event_times(
            t_refs=t_refs,
            t_before=t_before,
            t_after=t_after,
            description=description,
            global_t_ref=global_t_ref,
            time_unit=time_unit,
        )

    # Initialize lists to collect data
    annotated_data = data.copy()
    epochs = pd.DataFrame(
        columns=[
            "epoch id",
            "t_ref",
            "t_before",
            "t_after",
            "description",
            "epoch data",
        ]
    )

    # Iterate over each event time to create epochs
    for i, row in event_times.iterrows():
        t_ref_i = row["t_ref"]
        t_before_i = row["t_before"]
        t_after_i = row["t_after"]
        description_i = row["description"]

        start_time = t_ref_i - t_before_i
        end_time = t_ref_i + t_after_i
        mask = (data[ts_name] >= start_time) & (data[ts_name] <= end_time)

        if not mask.any():
            continue

        annotated_data.loc[mask, "epoch id"] = i
        annotated_data.loc[mask, "description"] = description_i
        annotated_data.loc[mask, "t_rel"] = data.loc[mask, ts_name] - t_ref_i

        local_data = data.loc[mask].copy()
        local_data["t_rel"] = local_data[ts_name] - t_ref_i
        local_data.reset_index(drop=True, inplace=True)

        epochs.at[i, "epoch id"] = i
        epochs.at[i, "t_ref"] = t_ref_i
        epochs.at[i, "t_before"] = t_before_i
        epochs.at[i, "t_after"] = t_after_i
        epochs.at[i, "description"] = description_i
        epochs.at[i, "epoch data"] = local_data

    # Drop rows where 'epoch data' is empty
    epochs = epochs.dropna(subset=["epoch data"]).reset_index(drop=True)

    # Set data types of the columns
    epochs = epochs.astype(
        {
            "epoch id": "Int32",
            "t_ref": "Int64",
            "t_before": "float",
            "t_after": "float",
            "description": "str",
            "epoch data": "object",
        }
    )
    annotated_data = annotated_data.astype(
        {"epoch id": "Int32", "description": "str", "t_rel": "float"}
    )

    return epochs, annotated_data


def extract_event_times(
    event_data: pd.DataFrame,
    t_before: float,
    t_after: float,
    event_name: str = "all",
) -> pd.DataFrame:
    """
    Construct event times from a list or array of reference times.

    Parameters
    ----------
    t_refs : list or np.ndarray
        List or array of reference times. Units specified by `time_unit`.
    t_before : float, np.ndarray, or list
        Time before the reference time to start the epoch, in **seconds**.
    t_after : float, np.ndarray, or list
        Time after the reference time to end the epoch, in **seconds**.
    description : str, np.ndarray, or list
        Description or label associated with the epoch.
    global_t_ref : int or float, optional
        Global reference time to be added to each reference time in `t_refs`. Units specified by `time_unit`. Default is 0.
    time_unit : str, optional
        Unit of time for the reference times and `global_t_ref` ('ns' for nanoseconds or 's' for seconds). Default is 'ns'.

    Returns
    -------
    event_times : pd.DataFrame
        DataFrame containing the constructed event times with columns:
        - 't_ref': Reference time of the event, in nanoseconds.
        - 't_before': Time before the reference time to start the epoch, in nanoseconds.
        - 't_after': Time after the reference time to end the epoch, in nanoseconds.
        - 'description': Description or label associated with the event.

    Notes
    -----
    - The `t_refs` and `global_t_ref` are combined and converted to nanoseconds according to `time_unit`.
    - The `t_before` and `t_after` parameters are always expected in **seconds** and will be converted to nanoseconds internally.
    """

    if "start timestamp [ns]" not in event_data.columns:
        raise ValueError("Data must contain a 'start timestamp [ns]' column.")

    description = None

    # check if the event_data has the 'type' column. It should only be present in concat streams
    if "type" in event_data.columns:
        data_type = "concat_steam"
    else:
        data_type = "single_stream"

    # check of event_data has fixation id column
    if data_type == "single_stream":
        if "fixation id" in event_data.columns:
            description = "fixation"
        elif "saccade id" in event_data.columns:
            description = "saccade"
        elif "blink id" in event_data.columns:
            description = "blink"

        event_times = event_data[["start timestamp [ns]"]].copy()
        event_times.rename(columns={"start timestamp [ns]": "t_ref"}, inplace=True)
        event_times["t_before"] = t_before
        event_times["t_after"] = t_after
        event_times["description"] = description

    # Extract event times
    if data_type == "concat_stream":
        if event_name == "all":
            event_data = event_data
            event_times = event_data[["start timestamp [ns]", "type"]].copy()
            event_times.rename(columns={"type": "description"}, inplace=True)

        elif (
            event_name == "fixation" or event_name == "saccade" or event_name == "blink"
        ):
            event_data = event_data[event_data["type"] == event_name]
            event_times = event_data[["start timestamp [ns]", "type"]].copy()
            event_times.rename(columns={"type": "description"}, inplace=True)

        else:
            event_data = event_data[event_data["message name"] == event_name]
            event_times = event_data[["start timestamp [ns]", "message name"]].copy()
            event_times.rename(columns={"message name": "description"}, inplace=True)

        event_times["t_before"] = t_before
        event_times["t_after"] = t_after

    return event_times


def construct_event_times(
    t_refs: Union[list, np.ndarray],
    t_before: Union[np.ndarray, float, None],
    t_after: Union[np.ndarray, float, None],
    description: Union[np.ndarray, str],
    global_t_ref: Union[int, float] = 0,
    time_unit: str = "ns",
) -> pd.DataFrame:
    """
    Construct event times from a list or array of reference times.

    Parameters
    ----------
    t_refs : list or np.ndarray
        List or array of reference times.
    t_before : float or np.ndarray
        Time before the reference time to start the epoch, in seconds.
    t_after : float or np.ndarray
        Time after the reference time to end the epoch, in seconds.
    description : str or np.ndarray
        Description or label associated with the epoch.
    global_t_ref : int or float, optional
        Global reference time to be added to each reference time, by default None.
    time_unit : str, optional
        Unit of time for the reference times ('ns' or 's'), by default "ns".

    Returns
    -------
    event_times : pd.DataFrame
        DataFrame containing the constructed event times.
    """

    # Check the input reference times
    if not isinstance(t_refs, (list, np.ndarray)):
        raise ValueError("t_refs must be a list or numpy array of reference times")

    n_epoch = len(t_refs)

    # Set conversion factor based on time unit
    if time_unit == "s":
        factor = 1e9  # Convert seconds to nanoseconds
    else:
        factor = 1  # Assume times are already in nanoseconds

    # Apply global_t_ref if provided
    if global_t_ref is not None:
        t_refs = [rt * factor + global_t_ref for rt in t_refs]

    # Handle t_before, t_after, and description
    other_info = []

    # Check each of the parameters (t_before, t_after, description) for correct length and type
    for x, name in zip(
        [t_before, t_after, description], ["t_before", "t_after", "description"]
    ):
        if isinstance(x, np.ndarray):
            # Ensure it's the same length as t_refs
            if len(x) != n_epoch:
                raise ValueError(f"{name} must have the same length as t_refs")
            other_info.append(x)
        elif isinstance(x, (int, float, str)):
            # If a single value or string is provided, repeat it for each epoch
            other_info.append(np.repeat(x, n_epoch))
        else:
            raise ValueError(f"{name} must be a single value, a list, or a numpy array")

    t_before, t_after, description = other_info

    # Construct the event times DataFrame
    event_times = pd.DataFrame(
        {
            "t_ref": t_refs,
            "t_before": t_before * factor,
            "t_after": t_after * factor,
            "description": description,
        }
    )

    return event_times
