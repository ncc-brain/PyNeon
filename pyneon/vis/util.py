
import cv2
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..video import SceneVideo

def plot_frame(
    video: "SceneVideo",
    index: int = 0,
    ax: Optional[plt.Axes] = None,
    auto_title: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a frame from the video on a matplotlib axis.

    Parameters
    ----------
    video : SceneVideo
        Video object to plot the frame from.
    index : int
        Index of the frame to plot.
    ax : matplotlib.axes.Axes or None
        Axis to plot the frame on. If ``None``, a new figure is created.
        Defaults to ``None``.
    auto_title : bool
        Whether to automatically set the title of the axis.
        The automatic title includes the video file name and the frame index.
        Defaults to ``True``.
    show : bool
        Show the figure if ``True``. Defaults to True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    ax : matplotlib.axes.Axes
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
            ax.set_title(f"{video.video_file.name} | Frame {index + 1}/{len(video)}")
        ax.axis("off")
    else:
        raise RuntimeError(f"Could not read frame {index}")
    if show:
        plt.show()

    video.reset()
    return fig, ax
