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
from .utils.variables import circular_columns
from .vis import plot_epochs


@fill_doc
class Epochs:
    """
    Class to create and analyze epochs in the data streams.

    Parameters
    ----------
    source : Stream or Events
        Data to create epochs from. Can be either a :class:`Stream` or
        a :class:`Events` instance.
    %(epochs_info)s

        Must not have empty values.

        See :func:`events_to_epochs_info` or :func:`construct_epochs_info`
        for helper functions to create this DataFrame.

    Notes
    -----
    An epoch spans the temporal range of ``t_ref - t_before`` to ``t_ref + t_after`` as shown below:

    ::

                        t_ref[0]                            t_ref[1]
            <--t_before[0]--|--t_after[0]-->      <-t_before[1]-|-t_after[1]->
        ├--------------------------------------------------------------------------------┤

    Attributes
    ----------
    %(epochs_info)s

        ======= ================================
        Column  Description
        ======= ================================
        t_start Start time of the epoch (``t_ref - t_before``).
        t_end   End time of the epoch (``t_ref + t_after``).
        ======= ================================

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
        self.epochs_info: pd.DataFrame = epochs_info
        self.source: Stream | Events = source.copy()
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
    def epochs_dict(self) -> dict[int, Stream | Events | None]:
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
            List of epoch indices for which no data was found.
        """
        return [
            int(epoch_index)
            for epoch_index, epoch in self.epochs_dict.items()
            if epoch is None
        ]

    def annotate_source(self) -> pd.DataFrame:
        """
        Create index-wise annotations of epoch indices for the source data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with index matching the source data indices and a column
            "epoch_indices" containing lists of epoch indices that include
            each data point.
        """
        source = self.source
        epochs_info = self.epochs_info

        # Timestamps from the source
        ts = source.ts if isinstance(source, Stream) else source.start_ts
        source_index = source.data.index
        annot = {i: [] for i in source_index}  # Initialize empty lists for each index

        # Iterate over each event time to create epochs
        empty_epochs = []
        for i, row in epochs_info.iterrows():
            t_ref_i, t_before_i, t_after_i = row[
                ["t_ref", "t_before", "t_after"]
            ].to_list()

            start_time = t_ref_i - t_before_i
            end_time = t_ref_i + t_after_i
            mask = np.logical_and(ts >= start_time, ts <= end_time)

            if not mask.any():
                empty_epochs.append(int(i))

            # Annotate the data with the epoch index
            for idx in source_index[mask]:
                annot[idx].append(int(i))

        if empty_epochs:
            warn(f"No data found for epoch(s): {empty_epochs}.", RuntimeWarning)

        annot_df = pd.DataFrame.from_dict(
            annot, orient="index", columns=["epoch index"]
        )
        return annot_df

    @property
    def t_ref(self) -> np.ndarray:
        """Reference time for each epoch in Unix nanoseconds.

        Returns
        -------
        numpy.ndarray
            Array of reference timestamps in nanoseconds.
        """
        return self.epochs_info["t_ref"].to_numpy()

    @property
    def t_before(self) -> np.ndarray:
        """Time before the reference time for each epoch in nanoseconds.

        Returns
        -------
        numpy.ndarray
            Array of time durations before reference in nanoseconds.
        """
        return self.epochs_info["t_before"].to_numpy()

    @property
    def t_after(self) -> np.ndarray:
        """Time after the reference time for each epoch in nanoseconds.

        Returns
        -------
        numpy.ndarray
            Array of time durations after reference in nanoseconds.
        """
        return self.epochs_info["t_after"].to_numpy()

    @property
    def description(self) -> np.ndarray:
        """Description or label for each epoch.

        Returns
        -------
        numpy.ndarray
            Array of description strings.
        """
        return self.epochs_info["description"].to_numpy()

    @property
    def columns(self) -> pd.Index:
        if self.data.empty:
            return pd.Index([])
        return self.data.columns.drop("epoch time [ns]", errors="ignore")

    @property
    def dtypes(self) -> pd.Series:
        """Data types of the epoched data."""
        if self.data.empty:
            return pd.Series(dtype=object)
        return self.data.drop(columns=["epoch time [ns]"], errors="ignore").dtypes

    @property
    def is_equal_length(self) -> bool:
        """Whether all epochs have the same length.

        Returns
        -------
        bool
            True if all epochs have identical t_before and t_after durations.
        """
        return np.allclose(self.t_before, self.t_before[0]) and np.allclose(
            self.t_after, self.t_after[0]
        )

    @property
    def has_overlap(self) -> bool:
        """Whether any adjacent epochs overlap in time.

        Returns
        -------
        bool
            True if any adjacent epochs have overlapping time intervals.
        """
        return self._check_overlap() != []

    @fill_doc
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
            Name of the column to plot for :class:`Epochs` constructed
            from a :class:`Stream`. If :class:`Epochs` was constructed
            from a :class:`Events`, this parameter is ignored. Defaults to None.
        cmap_name : str
            Colormap to use for different epochs. Defaults to 'cool'.
        %(ax_param)s
        %(show_param)s

        Returns
        -------
        %(fig_ax_returns)s
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
        Converts epochs into a 3D arrays with dimensions (n_epochs, n_channels, n_times).
        Acts similarly as :meth:`mne.Epochs.get_data`.

        Requires the epoch to be created from a :class:`Stream`.

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
        %(interp_kind_params)s

        Returns
        -------
        numpy.ndarray
            NumPy array of shape (n_epochs, n_channels, n_times).
        info : dict
            A dictionary containing:

            ============ ================================
            epoch_times  The common time grid, in nanoseconds.
            column_names List of provided column names.
            nan_flag     Boolean indicating whether NaN values were found in the data.
            ============ ================================
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
                max_gap_ms=None,
                inplace=False,
            ).data[column_names]
            epochs_np[i, :, :] = epoch_data.to_numpy().T

        info = {
            "epoch_times": epoch_times,
            "column_names": column_names,
            "nan_flag": np.isnan(epochs_np).any(),
        }

        return epochs_np, info

    def apply_baseline(
        self,
        baseline: tuple[Number | None, Number | None] = (None, 0),
        method: Literal["mean", "regression"] = "mean",
        exclude_cols: list[str] = [],
        inplace: bool = True,
    ) -> dict | None:
        """
        Apply baseline correction to epochs. Only applied to columns of float type.

        The baseline data is extracted and used to compute the correction.
        When ``method="mean"``, the mean of the baseline window is subtracted from the entire epoch.
        When ``method="regression"``, a linear trend is fitted to the baseline window and subtracted.

        For columns containing circular data (e.g., "yaw [deg]"), the correction is applied on unwrapped
        data (rad) and the result is wrapped back to degrees after correction.

        Parameters
        ----------
        baseline : tuple[Number or None, Number or None], optional
            Time window (relative to reference) for baseline computation in seconds.
            Defaults to (None, 0), which uses all data before the reference time.
        method : {"mean", "regression"}, optional
            Baseline correction method. Defaults to "mean".
        exclude_cols : list of str, optional
            Columns to exclude from baseline correction. Defaults to [].
        inplace : bool, optional
            If ``True``, replace :attr:`epochs_dict`. Otherwise returns a new instance of dict.
            Defaults to ``True``.

        Returns
        -------
        dict or None
            A new dict with modified epoch data if ``inplace=False``, otherwise ``None``.
            See :attr:`epochs_dict` for details.
        """

        def _get_baseline_mask(
            t_rel_sec: np.ndarray,
            t_min: Number | None,
            t_max: Number | None,
        ) -> np.ndarray:
            """Create boolean mask for baseline window."""
            if t_min is None:
                return t_rel_sec <= t_max
            elif t_max is None:
                return t_rel_sec >= t_min
            else:
                return (t_rel_sec >= t_min) & (t_rel_sec <= t_max)

        def _apply_baseline_correction(
            epoch_df: pd.DataFrame,
            cols_to_correct: list[str],
            circ_cols: list[str],
            baseline_mask: np.ndarray,
            epoch_time_s: np.ndarray,
            method: str,
        ) -> None:
            """Consolidated baseline correction for linear and circular float columns."""
            if method == "mean":
                baseline_means = epoch_df.loc[baseline_mask, cols_to_correct].mean()
                epoch_df.loc[:, cols_to_correct] -= baseline_means
            elif method == "regression":
                t_base = epoch_time_s[baseline_mask]
                for col in cols_to_correct:
                    y = epoch_df.loc[baseline_mask, col].to_numpy()

                    if (
                        len(t_base) < 2
                        or np.any(np.isnan(t_base))
                        or np.any(np.isnan(y))
                    ):
                        continue

                    # Fit trend on baseline and subtract from the whole trial
                    coeffs = np.polyfit(t_base, y, 1)
                    trend = np.polyval(coeffs, epoch_time_s)
                    epoch_df.loc[:, col] -= trend

            # Wrap circular columns back to range after correction
            for col in circ_cols:
                vals = epoch_df[col].to_numpy()
                vals_rad = vals * (2 * np.pi / 360)
                valid = ~np.isnan(vals_rad)
                vals_unwrapped_rad = np.full_like(vals_rad, np.nan)
                vals_unwrapped_rad[valid] = np.unwrap(vals_rad[valid])
                vals_deg_unwrapped = vals_unwrapped_rad * (360 / (2 * np.pi))
                vals_deg_wrapped = ((vals_deg_unwrapped + 180) % 360) - 180
                epoch_df.loc[:, col] = vals_deg_wrapped

        if not isinstance(self.source, Stream):
            raise TypeError("Baseline correction requires the source to be a Stream.")

        # Parse parameters
        t_min, t_max = baseline
        if t_min is not None and t_max is not None and (t_max < t_min):
            raise ValueError("baseline[1] must be >= baseline[0]")

        # Determine target streams/data
        if inplace:
            epochs_to_process = self.epochs_dict
        else:
            epochs_to_process = self.epochs_dict.copy()

        # Process each epoch
        for idx, epoch in epochs_to_process.items():
            if epoch is None:
                continue

            epoch_df = epoch.data.copy()
            # Only apply to float columns and respect excludes
            cols_to_correct = [
                c
                for c in epoch_df.select_dtypes(include=[float]).columns
                if c not in exclude_cols
            ]

            if not cols_to_correct:
                continue

            # Get baseline mask
            epoch_time_s = epoch_df["epoch time [ns]"].to_numpy() * 1e-9
            baseline_mask = _get_baseline_mask(epoch_time_s, t_min, t_max)

            if not baseline_mask.any():
                warn(f"Baseline window is empty for epoch {idx}.", RuntimeWarning)
                continue

            # Identify which target float columns are circular
            epoch_circ_cols = [c for c in cols_to_correct if c in circular_columns]

            # Step 2: Apply baseline correction (Linear or Mean) to all float columns
            _apply_baseline_correction(
                epoch_df,
                cols_to_correct,
                epoch_circ_cols,
                baseline_mask,
                epoch_time_s,
                method,
            )
            # Assign corrected data back to the epoch
            epoch.data = epoch_df
        return None if inplace else epochs_to_process


@fill_doc
def events_to_epochs_info(
    events: "Events",
    t_before: Number,
    t_after: Number,
    t_unit: Literal["s", "ms", "us", "ns"] = "s",
    event_name: str | list[str] = "all",
) -> pd.DataFrame:
    """
    Construct a ``epochs_info`` DataFrame suitable for creating epochs around event onsets.

    For simple event classes ("blinks", "fixations", "saccades"), all events
    in the input are used automatically. For more complex or combined event collections
    (e.g., loaded from ``events.csv``), you can either include all events
    (`event_name="all"`) or filter by specific names using ``event_name``.

    Parameters
    ----------
    events : Events
        Events instance containing the event times.
    t_before : numbers.Number
        Time before each event start to begin the epoch.
        Interpreted according to ``t_unit``.
    t_after : numbers.Number
        Time after each event start to end the epoch.
        Interpreted according to ``t_unit``.
    t_unit : str, optional
        Unit of time for ``t_before`` and ``t_after``.
        Can be "s", "ms", "us", or "ns". Defaults to "s".
    event_name : str or list of str, optional
        Only used if ``events.type`` is not one of "blinks", "fixations", or "saccades".
        Otherwise, ``events.data`` must have a ``name`` column indicating event labels.
        If `"all"`, all events from ``events.data`` are included,
        and their ``name`` values become the epoch descriptions.
        If a string or list is provided, only matching events are included.
        Defaults to "all".

    Returns
    -------
    %(epochs_info)s

    Examples
    --------
    Create ``epochs_info`` from blink events:

    >>> epochs_info = events_to_epochs_info(blinks, t_before=1, t_after=1)
    >>> print(epochs_info.head())
                     t_ref    t_before     t_after description
    0  1766068460987724691  1000000000  1000000000       blink
    1  1766068462919464691  1000000000  1000000000       blink
    2  1766068463785334691  1000000000  1000000000       blink
    3  1766068464836328691  1000000000  1000000000       blink
    4  1766068465932322691  1000000000  1000000000       blink

    Create ``epochs_info`` from "flash onset" events:

    >>> epochs_info = events_to_epochs_info(
        events, t_before=0.5, t_after=3, event_name="flash onset")
    >>> print(epochs_info.head())
                     t_ref   t_before     t_after  description
    0  1766068461745390000  500000000  3000000000  flash onset
    1  1766068465647497000  500000000  3000000000  flash onset
    2  1766068469642822000  500000000  3000000000  flash onset
    3  1766068473635128000  500000000  3000000000  flash onset
    4  1766068477629326000  500000000  3000000000  flash onset
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


@fill_doc
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
    Construct the ``epochs_info`` DataFrame for creating epochs. It populates
    single values for ``t_before``, ``t_after``, and ``description`` to match the length of ``t_ref``
    and converts all times to Unix timestamps in nanoseconds.

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
        (for instance :attr:`Stream.first_ts`).

    Returns
    -------
    %(epochs_info)s
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
