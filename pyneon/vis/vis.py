import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING, Union, Literal

if TYPE_CHECKING:
    from ..recording import NeonRecording
    from ..video import NeonVideo


def plot_frame(
    video: "NeonVideo",
    index: int = 0,
    ax: Union[plt.Axes, None] = None,
    auto_title: bool = True,
    show: bool = True,
):
    """
    Plot a frame from the video on a matplotlib axis.

    Parameters
    ----------
    video : :class:`NeonVideo`
        Video object to plot the frame from.
    index : int
        Index of the frame to plot.
    ax : :class:`matplotlib.pyplot.Axes` or None
        Axis to plot the frame on. If ``None``, a new figure is created.
        Defaults to ``None``.
    auto_title : bool
        Whether to automatically set the title of the axis.
        The automatic title includes the video file name and the frame index.
        Defaults to ``True``.

    Returns
    -------
    fig : :class:`matplotlib.pyplot.Figure`
        Figure object containing the plot.
    ax : :class:`matplotlib.pyplot.Axes`
        Axis object containing the plot.
    """
    if index >= len(video.ts) or index < 0:
        raise IndexError(f"Frame index {index} out of range")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = video.read()
    if ret:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if auto_title:
            ax.set_title(f"{video.video_file.name} | Frame {index+1}/{len(video)}")
        ax.axis("off")
    else:
        raise RuntimeError(f"Could not read frame {index}")
    if show:
        plt.show()
    return fig, ax


def plot_distribution(
    rec: "NeonRecording",
    heatmap_source: Literal["gaze", "fixations", None] = "gaze",
    scatter_source: Literal["gaze", "fixations", None] = "fixations",
    step_size: int = 10,
    sigma: Union[float, None] = 2,
    width_height: tuple[int, int] = (1600, 1200),
    cmap: Union[str, None] = "inferno",
    ax: Union[plt.Axes, None] = None,
    show: bool = True,
):
    """
    Plot a heatmap of gaze or fixation data on a matplotlib axis.
    Users can flexibly choose to generate a smoothed heatmap and/or scatter plot and
    the source of the data (gaze or fixation).

    Parameters
    ----------
    rec : :class:`NeonRecording`
        Recording object containing the gaze and video data.
    heatmap_source : {'gaze', 'fixations', None}
        Source of the data to plot as a heatmap. If None, no heatmap is plotted.
        Defaults to 'gaze'.
    scatter_source : {'gaze', 'fixations', None}
        Source of the data to plot as a scatter plot. If None, no scatter plot is plotted.
        Defaults to 'fixations'. Gaze data is typically more dense and thus less suitable
        for scatter plots.
    step_size : int
        Size of the grid cells in pixels. Defaults to 10.
    sigma : float or None
        Standard deviation of the Gaussian kernel used to smooth the heatmap.
        If None or 0, no smoothing is applied. Defaults to 2.
    width_height : tuple[int, int]
        If video is not available, the width and height of the scene camera frames to
        specify the heatmap dimensions. Defaults to (1600, 1200).
    cmap : str or None
        Colormap to use for the heatmap. Defaults to 'inferno'.
    ax : :class:`matplotlib.pyplot.Axes` or None
        Axis to plot the frame on. If ``None``, a new figure is created.
        Defaults to ``None``.
    show : bool
        Show the figure if ``True``. Defaults to True.

    Returns
    -------
    fig : :class:`matplotlib.pyplot.Figure`
        Figure object containing the plot.
    ax : :class:`matplotlib.pyplot.Axes`
        Axis object containing the plot.
    """
    if heatmap_source is None and scatter_source is None:
        raise ValueError(
            "At least one of heatmap_source and scatter_source must be provided."
        )

    if rec.video is None:
        if width_height is None:
            raise ValueError("No video data available and no width_height provided.")
        width_height = width_height
    else:
        width_height = (rec.video.width, rec.video.height)

    if rec.gaze is None:
        raise ValueError("No gaze data available.")
    if rec.fixations is None:
        raise ValueError("No fixation data available.")
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
