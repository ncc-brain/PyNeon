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
