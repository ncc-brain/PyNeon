from typing import TYPE_CHECKING, Literal, Optional

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter

from ..utils.doc_decorators import fill_doc
from ..video.marker import marker_family_to_dict
from ..video.utils import _validate_marker_layout

if TYPE_CHECKING:
    from ..epochs import Epochs
    from ..recording import Recording


@fill_doc
def plot_marker_layout(
    marker_layout: pd.DataFrame,
    show_marker_names: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Plot a 2D marker layout on a matplotlib axis.

    This renders the marker images on a white canvas using the marker metadata
    in ``marker_layout`` and overlays marker names at their centers.

    Parameters
    ----------
    %(marker_layout)s
    show_marker_names : bool
        Whether to overlay marker names at their centers. Defaults to True.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_returns)s
    layout: np.ndarray
        The rendered marker layout as a 2D numpy array (grayscale image).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Calculate actual bounds for each marker, then find overall canvas size
    _validate_marker_layout(marker_layout)
    min_x = (marker_layout["center x"] - marker_layout["size"] / 2).min()
    max_x = (marker_layout["center x"] + marker_layout["size"] / 2).max()
    min_y = (marker_layout["center y"] - marker_layout["size"] / 2).min()
    max_y = (marker_layout["center y"] + marker_layout["size"] / 2).max()

    layout_width = int(max_x - min_x)
    layout_height = int(max_y - min_y)

    # Create white background
    layout = Image.new("L", (layout_width, layout_height), color=255)

    for _, marker in marker_layout.iterrows():
        marker_name = marker["marker name"]
        marker_family = marker_name.rsplit("_", 1)[0]
        marker_id = int(marker_name.rsplit("_", 1)[1])

        marker_type, aruco_dict = marker_family_to_dict(marker_family)
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, int(marker["size"])
        )
        if marker_type == "april":
            # flip image diagonally for AprilTag markers
            # because AprilTag has a different orientation than ArUco
            # see https://github.com/opencv/opencv-python/issues/1195
            marker_img = np.flipud(np.fliplr(marker_img))

        # Convert to PIL Image
        marker_pil = Image.fromarray(marker_img)

        # Calculate paste position (top-left corner)
        x = int(marker["center x"] - marker["size"] / 2 - min_x)
        y = int(marker["center y"] - marker["size"] / 2 - min_y)

        # Paste marker onto canvas
        layout.paste(marker_pil, (x, y))

        if show_marker_names:
            ax.text(
                marker["center x"],
                marker["center y"],
                marker_name,
                color="magenta",
                ha="center",
                va="center",
            )

    # Display the canvas
    ax.imshow(layout, cmap="gray", extent=[min_x, max_x, max_y, min_y], origin="upper")
    ax.set_xlabel("x [surface coord]")
    ax.set_ylabel("y [surface coord]")
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax, np.array(layout)


@fill_doc
def plot_distribution(
    rec: "Recording",
    heatmap_source: Literal["gaze", "fixations", None] = "gaze",
    scatter_source: Literal["gaze", "fixations", None] = "fixations",
    step_size: int = 10,
    sigma: int | float = 2,
    width_height: tuple[int, int] = (1600, 1200),
    cmap: str = "inferno",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of gaze or fixation data on a matplotlib axis.
    Users can flexibly choose to generate a smoothed heatmap and/or scatter plot and
    the source of the data (gaze or fixation).

    Parameters
    ----------
    rec : Recording
        Recording instance containing the gaze and video data.
    heatmap_source : {'gaze', 'fixations', None}
        Source of the data to plot as a heatmap. If None, no heatmap is plotted.
        Defaults to 'gaze'.
    scatter_source : {'gaze', 'fixations', None}
        Source of the data to plot as a scatter plot. If None, no scatter plot is plotted.
        Defaults to 'fixations'. Gaze data is typically more dense and thus less suitable
        for scatter plots.
    step_size : int
        Size of the grid cells in pixels. Defaults to 10.
    sigma : int or float
        Standard deviation of the Gaussian kernel used to smooth the heatmap.
        If None or 0, no smoothing is applied. Defaults to 2.
    width_height : tuple[int, int]
        If video is not available, the width and height of the scene camera frames to
        specify the heatmap dimensions. Defaults to (1600, 1200).
    cmap : str
        Colormap to use for the heatmap. Defaults to 'inferno'.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_returns)s
    """
    if heatmap_source is None and scatter_source is None:
        raise ValueError(
            "At least one of heatmap_source and scatter_source must be provided."
        )

    try:
        scene_video = rec.scene_video
    except Exception:
        scene_video = None

    if scene_video is None:
        if width_height is None:
            raise ValueError("No video data available and no width_height provided.")
    else:
        width_height = (scene_video.width, scene_video.height)

    gaze = rec.gaze
    fixations = rec.fixations

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Generate the heatmap data
    x_edges = np.linspace(0, width_height[0], width_height[0] // step_size + 1)
    y_edges = np.linspace(0, width_height[1], width_height[1] // step_size + 1)

    # Sample gaze and fixation data (replace with your actual data)
    gaze_x = gaze.data["gaze x [px]"]
    gaze_y = gaze.data["gaze y [px]"]
    fix_x = fixations.data["fixation x [px]"]
    fix_y = fixations.data["fixation y [px]"]

    heatmap_x = gaze_x if heatmap_source == "gaze" else fix_x
    heatmap_y = gaze_y if heatmap_source == "gaze" else fix_y
    scatter_x = gaze_x if scatter_source == "gaze" else fix_x
    scatter_y = gaze_y if scatter_source == "gaze" else fix_y

    heatmap, _, _ = np.histogram2d(heatmap_x, heatmap_y, bins=(x_edges, y_edges))
    if sigma is not None or sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    if heatmap_source is not None:
        h = ax.imshow(
            heatmap.T,
            cmap=cmap,
            extent=[0, width_height[0], 0, width_height[1]],
            origin="lower",
        )
        plt.colorbar(h, ax=ax, label="Density")
    if scatter_source is not None:
        ax.scatter(scatter_x, scatter_y, color="white", s=10, alpha=0.3, edgecolors="k")
    ax.set_xlim(0, width_height[0])
    ax.set_ylim(0, width_height[1])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Scene camera x [px]")
    ax.set_ylabel("Scene camera y [px]")
    if show:
        plt.show()

    return fig, ax


@fill_doc
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
        Epochs instance containing the data to plot.
    column_name : str
        Name of the column to plot for :class:`pyneon.Epochs` constructed
        from a :class:`pyneon.Stream`. If :class:`pyneon.Epochs` was constructed
        from a :class:`pyneon.Events`, this parameter is ignored. Defaults to None.
    cmap_name : str
        Colormap to use for different epochs. Defaults to 'cool'.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_returns)s
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    num_epochs = len(epochs.epochs_dict)
    cmap = cm.get_cmap(cmap_name, num_epochs)
    norm = Normalize(vmin=0, vmax=num_epochs - 1)

    ax.axvline(0, color="k", linestyle="--")

    from pyneon import Events, Stream  # for source class check

    if isinstance(epochs.source, Stream):
        _plot_stream_epochs(epochs.epochs_dict, column_name, cmap, norm, fig, ax)
        ax.set_ylabel(column_name)
    elif isinstance(epochs.source, Events):
        _plot_event_epochs(epochs.epochs_dict, ax)
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


def _plot_stream_epochs(epochs, column_name, cmap, norm, fig, ax):
    """
    Internal helper to plot Stream-based Epochs.
    """
    if column_name is None:
        raise ValueError("Column name must be provided for Stream-based Epochs.")
    # Iterate through keys and values of the epochs.epochs dictionary
    for i, stream in epochs.items():
        if stream is None or stream.data.empty:
            continue  # Skip bad/missing data
        if column_name not in stream.columns:
            raise ValueError(f"Column '{column_name}' not found in epochs.")
        times = stream["epoch time [ns]"] / 1e9
        data = stream[column_name]
        color = cmap(norm(i))
        ax.plot(times, data, color=color)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch index")


def _plot_event_epochs(epochs, ax):
    """
    Internal helper to plot Events-based Epochs.
    """
    for i, events in epochs.items():
        if events is None:
            continue
        times = events.data["epoch time [ns]"] / 1e9

        # check if duration ms is in df
        if "duration [ms]" in events.data.columns:
            durations = events.data["duration [ms]"] / 1e3
            ax.hlines([i] * len(times), times, times + durations, color="gray")
        else:
            ax.scatter(times, [i] * len(times))
