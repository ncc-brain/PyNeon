# ruff: noqa: E402
from importlib.metadata import PackageNotFoundError, version

from .dataset import Dataset
from .epochs import Epochs, construct_epochs_info, events_to_epochs_info
from .events import Events
from .recording import Recording
from .stream import Stream
from .utils import *
from .video import Video, find_homographies
from .vis import plot_marker_layout

try:
    __version__ = version("pyneon")
except PackageNotFoundError:
    # Allows local source imports before the package is installed.
    __version__ = "0.0.0"

__all__ = [
    "Dataset",
    "Recording",
    "Stream",
    "Events",
    "Epochs",
    "Video",
    "plot_marker_layout",
    "find_homographies",
    "construct_epochs_info",
    "events_to_epochs_info",
]
