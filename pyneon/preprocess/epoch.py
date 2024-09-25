import pandas as pd
import numpy as np

from numbers import Number
from typing import TYPE_CHECKING, Union


import numpy as np
import pandas as pd

class Epoch:
    def __init__(self, data: pd.DataFrame, times: pd.DataFrame):
        self.data = data
        self.times = times

        # Check if data is uniformly sampled
        ts_diff = data["timestamp [ns]"].diff().dropna().unique()
        self.uniform_data = len(ts_diff) == 1

        # Create epochs
        self.epochs, self.data = create_epoch(data, times)

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
            self.epochs["t_before"].nunique() == 1 and self.epochs["t_after"].nunique() == 1
        )
        if self.equal_length:
            self.window_length = self.epochs["t_before"].iloc[0] + self.epochs["t_after"].iloc[0]


    def to_numpy(self, sampling_rate=100):
        """
        Converts epochs into a NumPy array with dimensions (n_epochs, n_times, n_channels).
        Resamples epochs to a fixed sampling rate.
        """
        # Ensure there are epochs to process
        if len(self.epochs) == 0:
            raise ValueError("No epochs with data to convert to NumPy array.")

        # Remove epochs with empty data
        self.epochs = self.epochs[self.epochs['epoch data'].apply(len) > 0].reset_index(drop=True)
        n_epochs = len(self.epochs)

        # Define the common time grid
        t_before = self.epochs['t_before'].iloc[0]
        t_after = self.epochs['t_after'].iloc[0]
        total_duration = t_after + t_before
        n_times = int(total_duration /1e9 * sampling_rate) + 1
        common_times = np.linspace(-t_before, t_after, n_times)

        # Assume all epochs have the same data columns (excluding 't_rel')
        data_columns = self.epochs.iloc[0]['epoch data'].columns.drop('t_rel')
        n_channels = len(data_columns)

        # Initialize the NumPy array
        epochs_np = np.full((n_epochs, n_times, n_channels), np.nan)

        # Interpolate each epoch onto the common time grid
        for i, (_, epoch) in enumerate(self.epochs.iterrows()):
            epoch_data = epoch['epoch data'].copy()
            t_rel = epoch_data['t_rel'].values
            for idx, col in enumerate(data_columns):
                y = epoch_data[col].values
                # Interpolate using numpy.interp
                interp_values = np.interp(
                    common_times,
                    t_rel,
                    y,
                    left=np.nan,
                    right=np.nan
                )
                epochs_np[i, :, idx] = interp_values

        return epochs_np


    def __len__(self):
        return len(self.epochs)



def create_epoch(
    data: pd.DataFrame,
    times_df: Union[pd.DataFrame, None] = None,
    t_ref: Union[np.ndarray, None] = None,
    t_before: Union[np.ndarray, Number, None] = None,
    t_after: Union[np.ndarray, Number, None] = None,
    description: Union[np.ndarray, None] = None,
    ):
    """
    Create epochs in the data stream(s) based on the input epochs dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Data stream to create epochs from. Must contain a 'timestamp [ns]' column.
    times : pd.DataFrame
        DataFrame containing the epochs information with the following columns:
        - 't_ref': Reference time of the epoch in seconds.
        - 't_before': Time before the reference time to start the epoch, in seconds.
        - 't_after': Time after the reference time to end the epoch, in seconds.
        - 'description': Message or label associated with the epoch.

    Returns
    -------
    data : pd.DataFrame
        Data stream with appended ``epoch id``, ``'t_rel'``, and ``description``.
    epochs_data : pd.DataFrame
        DataFrame where each row corresponds to an epoch, containing the data belonging to the epoch as a nested DataFrame.
    """

    if "timestamp [ns]" in data.columns:
        ts_name = "timestamp [ns]"
    elif "start timestamp [ns]" in data.columns:
        ts_name = "start timestamp [ns]"
    else:
        raise ValueError("Data must contain a 'timestamp [ns]' or 'start timestamp [ns]' column.")
    
    if times_df is not None:
        # Ensure the DataFrame has the required columns
        if not all(col in times_df.columns for col in ['t_ref', 't_before', 't_after', 'description']):
            raise ValueError("DataFrame must contain 't_ref', 't_before', 't_after', and 'description' columns")
        # Extract the columns from the DataFrame
        t_ref = times_df['t_ref'].to_numpy()
        t_before = times_df['t_before'].to_numpy()
        t_after = times_df['t_after'].to_numpy()
        description = times_df['description'].to_numpy()
    else:
        # Ensure the input arrays are not None
        if any(x is None for x in [t_ref, t_before, t_after, description]):
            raise ValueError("t_ref, t_before, t_after, and description must be provided if times_df is None")
        n_epoch = len(t_ref)
        other_info = []
        for x in [t_before, t_after, description]:
            # If an array is provided, ensure it has the same length as t_ref
            if isinstance(x, np.ndarray):
                if len(x) != n_epoch:
                    raise ValueError("If a numpy array is provided, it must have the same length as t_ref")
                other_info.append(x)
            # If a single value is provided, populate the list with the same value for each epoch
            else:
                other_info.append(np.repeat(x, n_epoch))
        t_before, t_after, description = other_info

    # Initialize lists to collect data
    annotated_data = data.copy()
    epochs = pd.DataFrame(
        columns=["epoch id", "t_ref", "t_before", "t_after", "description", "epoch data"]
    )
    
    for i, (t_ref_i, t_before_i, t_after_i, description_i) in enumerate(zip(t_ref, t_before, t_after, description)):
        start_time = t_ref_i - t_before_i
        end_time = t_ref_i + t_after_i
        mask = (data[ts_name] >= start_time) & (data[ts_name] <= end_time)
        if mask.empty:
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

    # Drop rows where 'data' is empty
    for i, epoch in epochs.iterrows():
        if epoch["epoch data"].empty:
            epochs.drop(i, inplace=True)

    # set datatypes of the columns
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
    Extract the timestamps of the events from the data stream.

    Parameters
    ----------
    data : pd.DataFrame
        Data stream to extract event timestamps from. Must contain a 'timestamp [ns]' column.
    t_before : float
        Time before the event to start the epoch, in seconds.
    t_after : float
        Time after the event to end the epoch, in seconds.
    event_name : str, optional
        Name of the event to extract. If 'all', extract all events. Only relevant if a concat stream or event data is provided.
    Returns
    -------
    event_times : pd.DataFrame
        DataFrame containing the timestamps of the events.
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
    for x, name in zip([t_before, t_after, description], ["t_before", "t_after", "description"]):
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
