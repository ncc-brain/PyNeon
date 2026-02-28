"""
Video processing and analysis module.

This module provides the :class:`Video` class which wraps OpenCV's VideoCapture
to provide convenient video handling with frame timestamps and camera metadata.

The Video class exposes most common video operations as convenience methods.
For users needing more advanced OpenCV functionality, the underlying
:attr:`~Video._cap` (cv2.VideoCapture) object is accessible, though care should
be taken to avoid interfering with PyNeon's internal state management.

See :class:`Video` documentation for details on using the ``_cap`` attribute safely.
"""

from .detect_contour import detect_contour
from .estimate_pose import estimate_camera_pose
from .homography import find_homographies
from .marker import detect_markers
from .scanpath import estimate_scanpath
from .video import Video

__all__ = [
    "Video",
    "estimate_scanpath",
    "detect_markers",
    "detect_contour",
    "estimate_camera_pose",
    "find_homographies",
]
