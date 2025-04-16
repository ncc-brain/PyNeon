import pandas as pd


def _check_stream_data(data: pd.DataFrame) -> None:
    """
    Check if the data is in the correct format for a stream.
    """
    # Check if index name is timestamp [ns]
    if data.index.name != "timestamp [ns]":
        raise ValueError("Index name must be 'timestamp [ns]'")
    # Check if index is sorted
    if not data.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in increasing order")


def _check_event_data(data: pd.DataFrame) -> None:
    """
    Check if the data is in the correct format for an event.
    """
    # Check if index name is timestamp [ns]
    if (data.index.name != "timestamp [ns]") and (
        data.index.name != "start timestamp [ns]"
    ):
        raise ValueError(
            "Index name must be 'timestamp [ns]' or 'start timestamp [ns]'"
        )
    # Check if index is sorted
    if not data.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in increasing order")
    # Try to convert the index to int64
    try:
        data.index = data.index.astype("Int64")
    except:
        raise ValueError(
            "Event index must be in UTC time in ns and thus convertible to int64"
        )
