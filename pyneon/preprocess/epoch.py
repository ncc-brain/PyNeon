import pandas as pd
import numpy as np

from typing import TYPE_CHECKING, Union


class Epoch:
    def __init__(self, data: pd.DataFrame, times: pd.DataFrame):
        self.data = data
        self.times = times

        self.epochs, self.data = create_epoch(data, times)

        # check if all epochs have the same length
        data_len = self.epochs["data"].apply(lambda x: x.shape[0])
        self.min_len = data_len.min()
        self.max_len = data_len.max()

        if data_len.unique().shape[0] == 1:
            self.equal_samples = True
        else:
            self.equal_samples = False

        if self.epochs["t_ref"].diff().unique().shape[0] == 1:
            self.equal_dist = True
        else:
            self.equal_dist = False

        if (
            self.epochs["t_before"].unique().shape[0] == 1
            and self.epochs["t_after"].unique().shape[0] == 1
        ):
            self.equal_length = True
            self.window_length = (
                self.epochs["t_before"].unique()[0] + self.epochs["t_after"].unique()[0]
            )
        else:
            self.equal_length = False

    def to_np(self):
        # check if all epochs have the same length
        if not self.equal_samples:
            if not self.equal_length or not self.equal_dist:
                raise ValueError(
                    "Epochs must have the same length, reference time, and time before/after values to convert to numpy array."
                )
            else:
                # check for smallest and largest number of samples
                min_samples = self.min_len
                max_samples = self.max_len

                df = self.window_length / max_samples
                df_str = str(df) + "ns"
                """
                #check if difference is less than 10%
                if (max_samples - min_samples) / min_samples > 0.1:
                    raise ValueError("Epochs must have the same length, reference time, and time before/after values to convert to numpy array.")
            
                #resample epochs on t_rel to have the maximum length
                else:
                """
                for idx in range(len(self.epochs)):
                    epoch = self.epochs.iloc[idx]
                    # Convert t_rel to timedelta
                    epoch["data"]["t_rel"] = pd.to_timedelta(
                        epoch["data"]["t_rel"], unit="ns"
                    )
                    epoch["data"] = (
                        epoch["data"].resample(df_str, on="t_rel").mean().reset_index()
                    )
                    # Convert t_rel back to ns
                    epoch["data"]["t_rel"] = (
                        epoch["data"]["t_rel"].dt.total_seconds() * 1e9
                    )
                    self.epochs.iloc[idx] = epoch

        n_epoch = len(self.epochs)
        n_times = self.epochs.iloc[0]["data"].shape[0]
        n_data = self.epochs.iloc[0]["data"].shape[1]

        epochs_np = np.nan((n_epoch, n_times, n_data))

        for i in range(n_epoch):
            epoch = self.epochs.iloc[i]
            epochs_np[i, :, :] = epoch["data"].values

        return epochs_np


def create_epoch(
    data: pd.DataFrame,
    times: pd.DataFrame,
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
        Data stream with appended 'epoch ID', 'time difference to reference time', and 'description'.
    epochs_data : pd.DataFrame
        DataFrame where each row corresponds to an epoch, containing the data belonging to the epoch as a nested DataFrame.
    """

    # Check that data has 'timestamp [ns]' column
    if "timestamp [ns]" not in data.columns:
        if "start timestamp [ns]" in data.columns:
            data.rename(
                columns={"start timestamp [ns]": "timestamp [ns]"}, inplace=True
            )
        else:
            raise ValueError("Data must contain a 'timestamp [ns]' column.")

    # Initialize lists to collect data
    data_with_epochs = data.copy()
    epochs = pd.DataFrame(
        columns=["epoch ID", "t_ref", "t_before", "t_after", "description", "data"]
    )

    # Iterate over each epoch
    for idx, row in times.iterrows():
        # all data in nanoseconds
        t_ref = row["t_ref"]
        t_before = row["t_before"]
        t_after = row["t_after"]
        description = row["description"]
        epoch_id = idx  # Use the index as the epoch ID

        start_time = t_ref - t_before
        end_time = t_ref + t_after

        # Extract data within the time window
        mask = (data["timestamp [ns]"] >= start_time) & (
            data["timestamp [ns]"] <= end_time
        )

        if mask.empty:
            continue

        data_with_epochs.loc[mask, "epoch ID"] = epoch_id
        data_with_epochs.loc[mask, "description"] = description
        data_with_epochs.loc[mask, "t_rel"] = (
            data.loc[mask, "timestamp [ns]"].copy() - t_ref
        )

        local_data = data.loc[mask].copy()
        local_data["t_rel"] = local_data["timestamp [ns]"] - t_ref
        local_data.reset_index(drop=True, inplace=True)

        epochs.at[epoch_id, "epoch ID"] = epoch_id
        epochs.at[epoch_id, "t_ref"] = t_ref
        epochs.at[epoch_id, "t_before"] = t_before
        epochs.at[epoch_id, "t_after"] = t_after
        epochs.at[epoch_id, "description"] = description
        epochs.at[epoch_id, "data"] = local_data

    # Drop rows where 'data' is empty
    for i in range(len(epochs)):
        if epochs["data"][i].empty:
            epochs.drop(i, inplace=True)

    # set datatypes of the columns
    epochs = epochs.astype(
        {
            "epoch ID": "Int32",
            "t_ref": "Int64",
            "t_before": "float",
            "t_after": "float",
            "description": "str",
        }
    )
    data_with_epochs = data_with_epochs.astype(
        {"epoch ID": "Int32", "description": "str", "t_rel": "float"}
    )

    return epochs, data_with_epochs


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


# create a function to create event times from a list or array of ref times, a fixed t_before, t_after, and message, plus an optional global_t_ref
def construct_event_times(
    t_refs: Union[list, np.ndarray],
    t_before: float,
    t_after: float,
    description: str,
    global_t_ref: Union[int, float] = 0,
    time_unit: str = "ns",
) -> pd.DataFrame:
    """
    Construct event times from a list or array of reference times.

    Parameters
    ----------
    t_refs : list or np.ndarray
        List or array of reference times.
    t_before : float
        Time before the reference time to start the epoch, in seconds.
    t_after : float
        Time after the reference time to end the epoch, in seconds.
    description : str
        Description or label associated with the epoch.
    global_t_ref : int or float, optional
        Global reference time to be added to each reference time, by default None.

    Returns
    -------
    event_times : pd.DataFrame
        DataFrame containing the constructed event times.
    """
    if time_unit == "s":
        factor = 1e9
    else:
        factor = 1

    if global_t_ref is not None:
        t_refs = [rt * factor + global_t_ref for rt in t_refs]

    event_times = pd.DataFrame(
        {
            "t_ref": t_refs,
            "t_before": t_before * factor,
            "t_after": t_after * factor,
            "description": description,
        }
    )

    return event_times
