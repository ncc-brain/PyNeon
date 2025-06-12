import pandas as pd
import numpy as np

from numbers import Number
from typing import Literal, Optional
import warnings
import matplotlib.pyplot as plt

from .stream import Stream
from .events import Events
from .vis import plot_epochs


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
    source : Stream or Events
        Data to create epochs from.
    times_df : pandas.DataFrame, shape (n_epochs, 4), optional
        DataFrame containing epoch information with the following columns:

            ``t_ref``: Reference time of the epoch, in nanoseconds.\n
            ``t_before``: Time before the reference time to start the epoch, in nanoseconds.\n
            ``t_after``: Time after the reference time to end the epoch, in nanoseconds.\n
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

    def __init__(self, source: Stream | Events, times_df: pd.DataFrame):
        if times_df.isnull().values.any():
            raise ValueError("times_df should not have any empty values")

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

        if isinstance(source, Stream):
            self.source_class = Stream
            self.is_uniformly_sampled = source.is_uniformly_sampled
            self.sf = source.sampling_freq_effective
        elif isinstance(source, Events):
            self.source_class = Events
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

    def plot(
        self,
        column_names: str,
        cmap_name: str = "cool",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot data from a specified column from epochs on a matplotlib axis.

        Parameters
        ----------
        column_name : str
            Name of the column to plot.
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
            column_names,
            cmap_name,
            ax,
            show,
        )
        return fig_ax

    def to_numpy(
        self,
        column_names: str | list[str] = "all",
    ) -> tuple[np.ndarray, dict]:
        """
        Converts epochs into a 3D array with dimensions (n_epochs, n_channels, n_times).
        Acts similarly as :meth:`mne.Epochs.get_data`.
        Requires the epoch to be created from a uniformly-sampled :class:`pyneon.Stream`.

        Parameters
        ----------
        column_names : str or list of str, optional
            Column names to include in the NumPy array. If 'all', all columns are included.
            Only columns that can be converted to int or float can be included.
            Default is 'all'.

        Returns
        -------
        numpy_epochs : numpy.ndarray
            NumPy array of shape (n_epochs, n_channels, n_times).

        info : dict
            A dictionary containing:

                ``"column_ids"``: List of provided column names.\n
                ``"t_rel"``: The common time grid, in nanoseconds.\n
                ``"nan_flag"``: Boolean indicating whether NaN values were found in the data.

        Notes
        -----
        - The time grid (``t_rel``) is in nanoseconds.
        - If `NaN` values are present after interpolation, they are noted in ``nan_flag``.
        """
        if self.source_class != Stream or not self.is_uniformly_sampled:
            raise ValueError(
                "The source must be a uniformly-sampled Stream to convert to NumPy array."
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
            for col in columns:
                if col not in self.columns:
                    raise ValueError(f"Column '{col}' doesn't exist in the data.")

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

    def baseline_correction(
        self,
        baseline: tuple[Number | None, Number | None] = (None, 0),
        method: str = "mean",
        inplace: bool = True,
    ) -> pd.DataFrame:
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
            * "mean" - subtract the scalar mean of the baseline window.
            * "linear" - fit a first-order (y = a·t + b) model *within* the
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
                warnings.warn(
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
                        warnings.warn(
                            f"Skipping linear baseline correction for '{col}' due to insufficient or invalid data.",
                            RuntimeWarning,
                        )
                        continue
                    if np.all(t_base == t_base[0]):
                        warnings.warn(
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

        if inplace:
            return self.data  # type: ignore[return-value]
        else:
            return data_copy


def _create_epochs(
    source: Stream | Events, times_df: pd.DataFrame
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
    if isinstance(source, Stream):
        ts = source.ts
    elif isinstance(source, Events):
        ts = source.start_ts
    else:
        raise ValueError("Source must be a Stream or Events.")

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
    events: "Events",
    t_before: Number,
    t_after: Number,
    t_unit: Literal["s", "ms", "us", "ns"] = "s",
    event_name: str | list[str] = "all",
) -> pd.DataFrame:
    """
    Construct a ``times_df`` DataFrame suitable for creating epochs from event data.
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
        Can be ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``. Default is ``"s"``.
    event_name : str or list of str, optional
        Only used if ``events`` includes more than one event type.
        If ``"all"``, all events are used. Otherwise, the ``name`` column is used to filter events
        whose names are in the list. Default to ``"all"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ``t_ref``, ``t_before``, ``t_after``, ``description`` (all in ns).
    """
    if events.event_name in ["blinks", "fixations", "saccades"]:
        description = events.event_name[:-1]  # Remove the 's' at the end
        t_ref = events.start_ts
    else:
        if "name" not in events.data.columns:
            raise ValueError(
                "Event data must have a 'name' column to filter by event name."
            )

        names = events.data["name"].astype(str)
        if event_name == "all":
            t_ref = events.data.index.to_numpy()
            description = names.to_numpy()
        else:
            if isinstance(event_name, str):
                event_name = [event_name]
            mask = names.isin(event_name)
            if not mask.any():
                raise ValueError(f"No events found matching names: {event_name}")
            filtered_data = events.data[mask]
            t_ref = filtered_data.index.to_numpy()
            description = filtered_data["name"].to_numpy()

    times_df = construct_times_df(
        t_ref,
        t_before,
        t_after,
        description,
        "ns",
        t_unit,
    )
    return times_df


def construct_times_df(
    t_ref: np.ndarray,
    t_before: np.ndarray | Number,
    t_after: np.ndarray | Number,
    description: np.ndarray | str,
    t_ref_unit: Literal["s", "ms", "us", "ns"] = "ns",
    t_other_unit: Literal["s", "ms", "us", "ns"] = "s",
    global_t_ref: int = 0,
) -> pd.DataFrame:
    """
    Handles the construction of the ``times_df`` DataFrame for creating epochs. It populates
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
        Can be ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``. Default is ``"ns"``.
    t_other_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``.
        Can be ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``. Default is ``"s"``.
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
    times_df = times_df.astype(
        {
            "t_ref": "int64",
            "t_before": "int64",
            "t_after": "int64",
            "description": "str",
        }
    )
    return times_df
