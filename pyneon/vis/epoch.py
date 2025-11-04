
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..epochs import Epochs


def plot_epochs(
    epochs: "Epochs",
    column_name: Optional[str] = None,
    cmap_name: str = "cool",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data from a specified column from epochs on a matplotlib axis.

    Works for Epochs created from Stream (time series) or Events (point-based).

    Parameters
    ----------
    epochs : Epochs
        Epochs object containing the data to plot.
    column_name : str
        Name of the column to plot for :class:`pyneon.Epochs` constructed
        from a :class:`pyneon.Stream`. If :class:`pyneon.Epochs` was constructed
        from a :class:`pyneon.Events`, this parameter is ignored. Defaults to None.
    cmap_name : str
        Colormap to use for different epochs. Defaults to 'cool'.
    ax : matplotlib.axes.Axes or None
        Axis to plot the data on. If ``None``, a new figure is created.
    show : bool
        Show the figure if ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
        Axis object containing the plot.
    """
    from pyneon import Stream, Events  # for source class check

    if epochs.source_class == Stream:
        if column_name is None:
            raise ValueError("Column name must be provided for Stream-based Epochs.")
        if column_name not in epochs.columns:
            raise ValueError(f"Column '{column_name}' not found in epochs.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    num_epochs = len(epochs.epochs)
    cmap = cm.get_cmap(cmap_name, num_epochs)
    norm = Normalize(vmin=0, vmax=num_epochs - 1)

    ax.axvline(0, color="k", linestyle="--")

    if epochs.source_class == Stream:
        _plot_stream_epochs(fig, epochs, column_name, cmap, norm, ax)
        ax.set_ylabel(column_name)
    elif epochs.source_class == Events:
        _plot_event_epochs(epochs, norm, ax)
        ax.set_ylabel("Epoch index")
    else:
        raise ValueError(
            "Epochs must be created from a `Stream` or `Events` to be plotted."
        )

    ax.set_xlabel("Epoch time (s)")
    ax.set_xlim(-epochs.t_before.max() / 1e9, epochs.t_after.max() / 1e9)

    if show:
        plt.show()
    return fig, ax


def _plot_stream_epochs(fig, epochs, column_name, cmap, norm, ax):
    """
    Internal helper to plot Stream-based Epochs.
    """
    for i, row in epochs.epochs.iterrows():
        times = row.data["epoch time"] / 1e9
        data = row.data[column_name]
        color = cmap(norm(i))
        ax.plot(times, data, color=color, label=f"Epoch {i}")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch Index")


def _plot_event_epochs(epochs, norm, ax):
    """
    Internal helper to plot Events-based Epochs.
    """
    for i, row in epochs.epochs.iterrows():
        df = row.data
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue  # Skip bad/missing data
        times = df["epoch time"] / 1e9

        # check if duration ms is in df
        if "duration [ms]" in df.columns:
            durations = df["duration [ms]"] / 1e3
            ax.hlines([i] * len(times), times, times + durations, color="gray")
        else:
            ax.scatter(times, [i] * len(times), label=f"Epoch {i}")
