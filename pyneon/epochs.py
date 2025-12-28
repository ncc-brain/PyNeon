from functools import cached_property
from numbers import Number
from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .events import Events
from .stream import Stream
from .utils.doc_decorators import fill_doc
from .vis import plot_epochs


class Epochs:
    """
    Class to create and manage epochs in the data streams.

    Parameters
    ----------
    source : Stream or Events
        Data to create epochs from.
    epochs_info : pandas.DataFrame, shape (n_epochs, 4)
        DataFrame containing epoch information with the following columns (time in ns):

            ``t_ref``: Reference time of the epoch.\n
            ``t_before``: Time before the reference time to start the epoch.\n
            ``t_after``: Time after the reference time to end the epoch.\n
            ``description``: Description or label associated with the epoch.

        Must not have empty values.

    Notes
    -----
    An epoch spans the temporal range of ``t_ref - t_before`` to ``t_ref + t_after`` as shown below:

    ::

                        t_ref[0]                            t_ref[1]
            <--t_before[0]--|--t_after[0]-->      <-t_before[1]-|-t_after[1]->
        ├--------------------------------------------------------------------------------┤

    Attributes
    ----------
    epochs_info : pandas.DataFrame
        The supplied epochs information DataFrame with additional columns:

            ``t_start``: Start time of the epoch (``t_ref - t_before``).\n
            ``t_end``: End time of the epoch (``t_ref + t_after``).

    source : Stream or Events
        The source data used to create epochs.
    """

    def __init__(self, source: Stream | Events, epochs_info: pd.DataFrame):
        if epochs_info.empty or epochs_info.isnull().values.any():
            raise ValueError("epochs_info must not be empty or contain NaN values.")

        epochs_info = epochs_info.sort_values("t_ref").reset_index(drop=True)
        epochs_info.index.name = "epoch index"
        epochs_info["t_start"] = epochs_info["t_ref"] - epochs_info["t_before"]
        epochs_info["t_end"] = epochs_info["t_ref"] + epochs_info["t_after"]

        # Set columns to appropriate data types (check if columns are present along the way)
        epochs_info = epochs_info.astype(
            {
                "t_ref": "int64",
                "t_before": "int64",
                "t_after": "int64",
                "t_start": "int64",
                "t_end": "int64",
                "description": "str",
            }
        )
        self.epochs_info = epochs_info
        self.source = source.copy()
        self._check_overlap()

    def __len__(self):
        return self.epochs_info.shape[0]

    def _check_overlap(self) -> list[tuple[int, int] | None]:
        overlap_epochs = []
        for i in range(1, self.epochs_info.shape[0]):
            # Check if the current epoch overlaps with the previous epoch
            if (
                self.epochs_info["t_ref"].iloc[i] - self.epochs_info["t_before"].iloc[i]
                < self.epochs_info["t_ref"].iloc[i - 1]
                + self.epochs_info["t_after"].iloc[i - 1]
            ):
                overlap_epochs.append((i - 1, i))
        if overlap_epochs:
            warn(
                f"The following epochs overlap in time:\n{overlap_epochs}",
                RuntimeWarning,
            )
        return overlap_epochs

    @cached_property
    def epochs(self) -> dict[int, Stream | Events | None]:
        """
        Dictionary of epochs indexed by epoch index. Each epoch contains
        data cropped from the source between ``t_start`` and ``t_end``.
        If no data is found for an epoch, its value is ``None``.

        Returns
        -------
        dict of int to Stream or Events or None
            Dictionary mapping epoch indices to their corresponding data.
        """
        epochs = {}
        empty_epochs = []
        for epoch_index in self.epochs_info.index:
            t_ref = self.epochs_info.at[epoch_index, "t_ref"]
            t_start = self.epochs_info.at[epoch_index, "t_start"]
            t_end = self.epochs_info.at[epoch_index, "t_end"]
            try:
                epoch = self.source.crop(t_start, t_end, by="timestamp", inplace=False)
                ts = epoch.ts if isinstance(epoch, Stream) else epoch.start_ts
                epoch.data["epoch time [ns]"] = ts - t_ref
                epochs[int(epoch_index)] = epoch
            except ValueError:
                empty_epochs.append(int(epoch_index))
                epochs[int(epoch_index)] = None
        if empty_epochs:
            warn(f"No data found for epoch(s): {empty_epochs}.", RuntimeWarning)
        return epochs

    @property
    def empty_epochs(self) -> list[int]:
        """Indices of epochs that contain no data.

        Returns
        -------
        list of int
            List of epoch indices that are empty.
        """
        return [
            int(epoch_index)
            for epoch_index, epoch in self.epochs.items()
            if epoch is None
        ]

    @property
    def t_ref(self) -> np.ndarray:
        """The reference time for each epoch in UTC nanoseconds."""
        return self.epochs_info["t_ref"].to_numpy()

    @property
    def t_before(self) -> np.ndarray:
        """The time before the reference time for each epoch in nanoseconds."""
        return self.epochs_info["t_before"].to_numpy()

    @property
    def t_after(self) -> np.ndarray:
        """The time after the reference time for each epoch in nanoseconds."""
        return self.epochs_info["t_after"].to_numpy()

    @property
    def description(self) -> np.ndarray:
        """The description or label for each epoch."""
        return self.epochs_info["description"].to_numpy()

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
        return self._check_overlap() != []

    def plot(
        self,
        column_name: Optional[str] = None,
        cmap_name: str = "cool",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot data from a specified column from epochs on a matplotlib axis.

        Parameters
        ----------
        column_name : str
            Name of the column to plot for :class:`pyneon.Epochs` constructed
            from a :class:`pyneon.Stream`. If :class:`pyneon.Epochs` was constructed
            from a :class:`pyneon.Events`, this parameter is ignored. Defaults to None.
        cmap_name : str
            Colormap to use for different epochs. Defaults to 'cool'.
        ax : matplotlib.axes.Axes or None
            Axis to plot the data on. If ``None``, a new figure is created.
            Defaults to ``None``.
        show : bool
            Show the figure if ``True``. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.
        ax : matplotlib.axes.Axes
            Axis object containing the plot.
        """
        fig_ax = plot_epochs(
            self,
            column_name,
            cmap_name,
            ax,
            show,
        )
        return fig_ax

    @fill_doc
    def to_numpy(
        self,
        column_names: str | list[str] = "all",
        sampling_rate: Optional[Number] = None,
        float_kind: str | int = "linear",
        other_kind: str | int = "nearest",
    ) -> tuple[np.ndarray, dict]:
        """
        Converts epochs into a 3D array with dimensions (n_epochs, n_channels, n_times).
        Acts similarly as :meth:`mne.Epochs.get_data`.
        Requires the epoch to be created from a :class:`pyneon.Stream`.

        Parameters
        ----------
        column_names : str or list of str, optional
            Column names to include in the NumPy array. If "all", all columns are included.
            Only numerical columns can be included.
            Default to "all".
        sampling_rate : numbers.Number, optional
            Desired sampling rate in Hz for the output NumPy array.
            If None, the nominal sampling rate of the source Stream is used.
            Defaults to None.
        %(interp_kwargs)s

        Returns
        -------
        numpy.ndarray
            NumPy array of shape (n_epochs, n_channels, n_times).
        info : dict
            A dictionary containing:

                "column_ids": List of provided column names.\n
                "t_rel": The common time grid, in nanoseconds.\n
                "nan_flag": Boolean indicating whether NaN values were found in the data.
        """
        if not isinstance(self.source, Stream):
            raise TypeError("The source must be a Stream to convert to NumPy array.")
        if not self.is_equal_length:
            raise ValueError(
                "Epochs must have equal length (t_before and t_after) to convert to NumPy array."
            )
        sf = (
            self.source.sampling_freq_nominal
            if sampling_rate is None
            else sampling_rate
        )

        # Check if column names (str or list) are all in the source columns
        if column_names == "all":
            column_names = self.source.columns.to_list()
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            if col not in self.source.columns:
                raise ValueError(f"Column '{col}' not found in source Stream.")

        epoch_times = np.arange(
            -self.epochs_info["t_before"].iloc[0],
            self.epochs_info["t_after"].iloc[0],
            step=int(1e9 / sf),
            dtype="int64",
        )

        # Interpolate each epoch onto the common time grid
        epochs_np = np.full((len(self), len(column_names), len(epoch_times)), np.nan)
        for i, row in self.epochs_info.iterrows():
            t_ref = row["t_ref"]
            new_ts = epoch_times + t_ref
            epoch_data = self.source.interpolate(
                new_ts,
                float_kind=float_kind,
                other_kind=other_kind,
                inplace=False,
            ).data[column_names]
            epochs_np[i, :, :] = epoch_data.to_numpy().T

        info = {
            "epoch_times": epoch_times,
            "column_names": column_names,
            "nan_flag": np.isnan(epochs_np).any(),
        }

        return epochs_np, info

    def baseline_correction(
        self,
        baseline: tuple[Number | None, Number | None] = (None, 0),
        method: str = "mean",
        inplace: bool = True,
    ) -> dict[int, Stream | Events | None] | None:
        """
        Perform baseline correction on epochs.

        Parameters
        ----------
        baseline : (t_min, t_max), iterable of float | None
            Start and end of the baseline window **in seconds**, relative to
            the event trigger (t_ref = 0).  ``None`` means "from the first /
            up to the last sample".  Default: (None, 0.0) -> the pre-trigger
            part of each epoch.
        method : "mean" or "linear", optional
            * "mean" - Subtract the scalar mean of the baseline window.
            * "linear" - Fit a first-order (:math:`y = at + b`) model *within* the
              baseline window and remove the fitted trend from the entire
              epoch (a very small, fast version of MNE's regression detrending).

            Defaults to "mean".
        inplace : bool
            If True, overwrite epochs data.
            Otherwise return a **new, corrected** pandas.DataFrame
            and leave the object unchanged.
            Defaults to True.

        Returns
        -------

        pandas.DataFrame
            The baseline-corrected data (same shape & dtypes as original data).

        """
        if not isinstance(self.source, Stream):
            raise TypeError("Baseline correction requires the source to be a Stream.")

        def _fit_and_subtract(epoch_df: pd.DataFrame, chan_cols: list[str]) -> None:
            """In-place mean or linear detrend on *one* epoch DF."""
            # mask rows within the baseline window (epoch time is int64 ns)
            t_rel_sec = epoch_df["epoch time"].to_numpy() * 1e-9
            if t_min is None:
                mask = t_rel_sec <= t_max
            elif t_max is None:
                mask = t_rel_sec >= t_min
            else:
                mask = (t_rel_sec >= t_min) & (t_rel_sec <= t_max)

            if not mask.any():
                warn(
                    "Baseline window is empty for at least one epoch.",
                    RuntimeWarning,
                )
                return  # nothing to correct

            if method == "mean":
                baseline_mean = epoch_df.loc[mask, chan_cols].mean()
                epoch_df.loc[:, chan_cols] = epoch_df[chan_cols] - baseline_mean
            elif method == "linear":
                t_base = t_rel_sec[mask]
                for col in chan_cols:
                    y = epoch_df.loc[mask, col].to_numpy()

                    # Check for NaNs, length, and constant input
                    if (
                        len(t_base) < 2
                        or np.any(np.isnan(t_base))
                        or np.any(np.isnan(y))
                    ):
                        warn(
                            f"Skipping linear baseline correction for '{col}' due to insufficient or invalid data.",
                            RuntimeWarning,
                        )
                        continue
                    if np.all(t_base == t_base[0]):
                        warn(
                            f"Skipping linear baseline correction for '{col}' due to constant timestamps.",
                            RuntimeWarning,
                        )
                        continue

                    # Now it's safe to fit
                    a, b = np.polyfit(t_base, y, 1)
                    epoch_df.loc[:, col] = epoch_df[col] - (a * t_rel_sec + b)
            else:
                raise ValueError("method must be 'mean' or 'linear'")

        # ------------------------------------------------------------------
        # 1. Parse parameters
        # ------------------------------------------------------------------
        t_min, t_max = baseline
        if t_min is not None and t_max is not None and (t_max < t_min):
            raise ValueError("baseline[1] must be >= baseline[0]")

        chan_cols = self.columns.to_list()

        # Work on a copy unless the caller wants in-place modification
        if inplace:
            epochs_copy = self.epochs
            data_copy = self.data
        else:
            epochs_copy = self.epochs.copy(deep=True)
            data_copy = self.data.copy(deep=True)

        for idx, row in epochs_copy.iterrows():
            epoch_df: pd.DataFrame = row["data"]
            _fit_and_subtract(epoch_df, chan_cols)
            # write back (only needed when we are working on a *copy*)
            if not inplace:
                epochs_copy.at[idx, "data"] = epoch_df
                # update the global data DF as well
                mask = data_copy["epoch index"] == idx
                data_copy.loc[mask, chan_cols] = epoch_df[chan_cols].to_numpy()

        if not inplace:
            return data_copy


def annotate_epochs(source: Stream | Events, epochs_info: pd.DataFrame) -> dict:
    """
    Create index-wise annotations of epoch indices for the source data.
    """
    # _check_overlap(epochs_info)

    # Timestamps from the source
    ts = source.ts if isinstance(source, Stream) else source.start_ts
    source_index = source.data.index
    annot = {i: [] for i in source_index}  # Initialize empty lists for each index

    # Iterate over each event time to create epochs
    empty_epochs = []
    for i, row in epochs_info.iterrows():
        t_ref_i, t_before_i, t_after_i = row[["t_ref", "t_before", "t_after"]].to_list()

        start_time = t_ref_i - t_before_i
        end_time = t_ref_i + t_after_i
        mask = np.logical_and(ts >= start_time, ts <= end_time)

        if not mask.any():
            empty_epochs.append(i)

        # Annotate the data with the epoch index
        for idx in source_index[mask]:
            annot[idx].append(i)

    if empty_epochs:
        warn(f"No data found for epoch(s): {empty_epochs}.", RuntimeWarning)

    return annot


def events_to_epochs_info(
    events: "Events",
    t_before: Number,
    t_after: Number,
    t_unit: Literal["s", "ms", "us", "ns"] = "s",
    event_name: str | list[str] = "all",
) -> pd.DataFrame:
    """
    Construct a ``epochs_info`` DataFrame suitable for creating epochs from event data.
    For "simple" ``events`` (blinks, fixations, saccades), all events are used.
    For more complex ``events`` (e.g., from "events.csv", or concatenated events),
    the user can specify which events to include by a ``name`` column.

    Parameters
    ----------
    events : Events
        Events instance containing the event times.
    t_before : numbers.Number
        Time before the event start time to start the epoch. Units specified by ``t_unit``.
    t_after : numbers.Number
        Time after the event start time to end the epoch. Units specified by ``t_unit``.
    t_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``.
        Can be "s", "ms", "us", or "ns". Default is "s".
    event_name : str or list of str, optional
        Only used if ``events`` includes more than one event type.
        If "all", all events are used. Otherwise, the ``name`` column is used to filter events
        whose names are in the list. Default to "all".

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ``t_ref``, ``t_before``, ``t_after``, ``description`` (all in ns).
    """
    t_ref = events.start_ts
    if events.type in ["blinks", "fixations", "saccades"]:
        description = events.type[:-1]  # Remove the 's' at the end
    else:
        if "name" not in events.data.columns:
            raise ValueError(
                "Event data must have a 'name' column to filter by event name."
            )

        names = events.data["name"].astype(str)
        if event_name == "all":
            description = names.to_numpy()
        else:
            matching_events = events.filter_by_name(event_name)
            t_ref = matching_events.start_ts
            description = matching_events.data["name"].to_numpy()

    epochs_info = construct_epochs_info(
        t_ref,
        t_before,
        t_after,
        description,
        t_ref_unit="ns",
        t_other_unit=t_unit,
    )
    return epochs_info


def construct_epochs_info(
    t_ref: np.ndarray,
    t_before: np.ndarray | Number,
    t_after: np.ndarray | Number,
    description: np.ndarray | str,
    t_ref_unit: Literal["s", "ms", "us", "ns"] = "ns",
    t_other_unit: Literal["s", "ms", "us", "ns"] = "s",
    global_t_ref: int = 0,
) -> pd.DataFrame:
    """
    Handles the construction of the ``epochs_info`` DataFrame for creating epochs. It populates
    single values for `t_before`, `t_after`, and `description` to match the length of `t_ref`.
    and converts all times to UTC timestamps in nanoseconds.

    Parameters
    ----------
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
    t_ref_unit : str, optional
        Unit of time for ``t_ref``.
        Can be "s", "ms", "us", or "ns". Default is "ns".
    t_other_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``.
        Can be "s", "ms", "us", or "ns". Default is "s".
    global_t_ref : int, optional
        Global reference time (in nanoseconds) to be added to `t_ref`.
        Unit is nanosecond. Defaults to 0. This is useful when the reference times
        are relative to a global start time
        (for instance :attr:`pyneon.Stream.first_ts`).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ``t_ref``, ``t_before``, ``t_after``, ``description`` (all in ns).
    """

    if (n_epoch := len(t_ref)) == 0:
        raise ValueError("t_ref must not be empty")

    time_factors = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1}

    # Check each of the parameters (t_before, t_after, description) for correct length and type
    for x, name in zip(
        [t_before, t_after, description], ["t_before", "t_after", "description"]
    ):
        if isinstance(x, np.ndarray):
            # Ensure it's the same length as t_ref
            if len(x) != n_epoch:
                raise ValueError(
                    f"{name} must have the same length as t_ref ({n_epoch}), got {len(x)}"
                )
        elif isinstance(x, (Number, str)):
            x = np.repeat(x, n_epoch)
        else:
            raise ValueError(f"{name} must be a single value or a numpy array")

    # Construct the event times DataFrame
    # Do rounding as they should be timestamps already
    epochs_info = pd.DataFrame(
        {
            "t_ref": t_ref * time_factors[t_ref_unit] + global_t_ref,
            "t_before": t_before * time_factors[t_other_unit],
            "t_after": t_after * time_factors[t_other_unit],
            "description": description,
        }
    )
    epochs_info = epochs_info.astype(
        {
            "t_ref": "int64",
            "t_before": "int64",
            "t_after": "int64",
            "description": "str",
        }
    )
    return epochs_info
