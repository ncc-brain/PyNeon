import pandas as pd
from typing import Union
from numbers import Number


def time_to_ts(time: Number, data: pd.DataFrame):
    if "time [s]" not in data.columns or "timestamp [ns]" not in data.columns:
        raise ValueError('Dataframe must have columns "time [s]" and "timestamp [ns]"')
    ts = data.loc[(data["time [s]"] - time).abs().idxmin(), "timestamp [ns]"]
    return ts
