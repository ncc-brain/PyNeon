import pandas as pd
import numpy as np

from typing import TYPE_CHECKING, Union

def create_epoch(
                data: pd.DataFrame,
                times: pd.DataFrame,
                global_ref_time: Union[int, float] = None,
                time_unit: str = 'ns'
                ) -> pd.DataFrame:
    """
    Create epochs in the data stream(s) based on the input epochs dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Data stream to create epochs from. Must contain a 'timestamp [ns]' column.
    times : pd.DataFrame
        DataFrame containing the epochs information with the following columns:
        - 'ref_time': Reference time of the epoch in seconds.
        - 't_before': Time before the reference time to start the epoch, in seconds.
        - 't_after': Time after the reference time to end the epoch, in seconds.
        - 'message': Message or label associated with the epoch.

    Returns
    -------
    data : pd.DataFrame
        Data stream with appended 'epoch ID', 'time difference to reference time', and 'message'.
    epochs_data : pd.DataFrame
        DataFrame where each row corresponds to an epoch, containing the data belonging to the epoch as a nested DataFrame.
    """

    # Check that data has 'timestamp [ns]' column
    if 'timestamp [ns]' not in data.columns:
        raise ValueError("Data must contain a 'timestamp [ns]' column.")

    # Initialize lists to collect data
    data_with_epochs_list = []
    epochs_list = []

    if global_ref_time is not None:
        times['ref_time'] = times['ref_time'] + global_ref_time

    if time_unit == 's':
        factor = 1e9
    else:
        factor = 1

    # Iterate over each epoch
    for idx, row in times.iterrows():
        #all data in nanoseconds
        ref_time = row['ref_time'] * factor
        t_before = row['t_before'] * factor
        t_after = row['t_after'] * factor
        message = row['message']
        epoch_id = idx  # Use the index as the epoch ID

        start_time = ref_time - t_before
        end_time = ref_time + t_after

        # Extract data within the time window
        mask = (data['timestamp [ns]'] >= start_time) & (data['timestamp [ns]'] <= end_time)
        epoch_data = data.loc[mask].copy()

        if epoch_data.empty:
            continue  # No data in this epoch

        # Compute time difference to reference time (in seconds)
        epoch_data['time difference to reference time'] = (epoch_data['timestamp [ns]'] - ref_time)  # in seconds

        # Add 'epoch ID' and 'message'
        epoch_data['epoch ID'] = epoch_id
        epoch_data['message'] = message

        # Append to the data_with_epochs_list
        data_with_epochs_list._append(epoch_data)

        # For the epochs dataframe, store the epoch info and the data as a nested dataframe
        epoch_info = {
            'epoch ID': epoch_id,
            'ref_time': ref_time,
            't_before': t_before,
            't_after': t_after,
            'message': message,
            'data': epoch_data.reset_index(drop=True)  # Reset index for clarity
        }
        epochs_list._append(epoch_info)

    # Concatenate the data_with_epochs_list into a single dataframe
    if data_with_epochs_list:
        epochs = pd.concat(data_with_epochs_list, ignore_index=True)
    else:
        epochs = pd.DataFrame()  # Empty dataframe

    # Create the epochs dataframe
    data = pd.DataFrame(epochs_list)

    return epochs, data

def extract_event_times(
    event_data: pd.DataFrame,
    t_before: float,
    t_after: float,
    event_name: str = 'all',
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

    if 'start timestamp [ns]' not in event_data.columns:
        raise ValueError("Data must contain a 'start timestamp [ns]' column.")
    
    message = None

    # check if the event_data has the 'type' column. It should only be present in concat streams
    if 'type' in event_data.columns:
        data_type = 'concat_steam'
    else:
        data_type = 'single_stream'

    #check of event_data has fixation id column
    if data_type == 'single_stream':
        if 'fixation id' in event_data.columns:
            message = 'fixation'
        elif 'saccade id' in event_data.columns:
            message = 'saccade'
        elif 'blink id' in event_data.columns:
            message = 'blink'

        event_times = event_data[['start timestamp [ns]']].copy()
        event_times.rename(columns={'start timestamp [ns]': 'ref_time'}, inplace=True)
        event_times['t_before'] = t_before
        event_times['t_after'] = t_after
        event_times['message'] = message

    # Extract event times
    if data_type == 'concat_stream':
        if event_name == 'all':
            event_data = event_data
            event_times = event_data[['start timestamp [ns]', 'type']].copy()
            event_times.rename(columns={'type': 'message'}, inplace=True)

        elif event_name == 'fixation' or event_name == 'saccade' or event_name == 'blink':
            event_data = event_data[event_data['type'] == event_name]
            event_times = event_data[['start timestamp [ns]', 'type']].copy()
            event_times.rename(columns={'type': 'message'}, inplace=True)

        else:
            event_data = event_data[event_data['message name'] == event_name]
            event_times = event_data[['start timestamp [ns]', 'message name']].copy()
            event_times.rename(columns={'message name': 'message'}, inplace=True)
        
        event_times['t_before'] = t_before
        event_times['t_after'] = t_after

    return event_times


