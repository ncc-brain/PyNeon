import cv2
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..recording import NeonRecording
    from ..video import NeonVideo


def plot_frame(
    video: "NeonVideo",
    index: int = 0,
    ax: Union[plt.Axes, None] = None,
    auto_title: bool = True,
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
    return fig, ax