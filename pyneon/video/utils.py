import re
from numbers import Number
from typing import Any, Iterable, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .constants import APRILTAG_FAMILIES, ARUCO_NUMBERS, ARUCO_SIZES


def _verify_format(df: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    """Verify that the DataFrame contains all expected columns (including index)."""
    actual_columns = set(df.columns)
    if df.index.name:
        actual_columns.add(df.index.name)

    missing = set(expected_columns) - actual_columns
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def get_undistort_maps(
    video: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached undistortion maps and the new camera matrix."""
    cache = getattr(video, "_undistort_cache", None)
    if cache is None:
        camera_matrix = video.camera_matrix
        dist_coeffs = video.distortion_coefficients
        frame_width, frame_height = video.width, video.height
        optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (frame_width, frame_height),
            1,
            (frame_width, frame_height),
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            optimal_camera_matrix,
            (frame_width, frame_height),
            cv2.CV_16SC2,
        )
        cache = (map1, map2, optimal_camera_matrix)
        setattr(video, "_undistort_cache", cache)
    return cache


def distort_points(video: Any, points: np.ndarray) -> np.ndarray:
    """Map undistorted pixel coordinates back to the original camera frame."""
    pts = np.asarray(points, dtype=np.float32)
    pts_2d = np.atleast_2d(pts)
    if pts_2d.size == 0:
        return pts_2d.reshape(-1, 2)

    _, _, new_camera_matrix = get_undistort_maps(video)
    fx, fy = new_camera_matrix[0, 0], new_camera_matrix[1, 1]
    cx, cy = new_camera_matrix[0, 2], new_camera_matrix[1, 2]
    normalized = np.column_stack(
        [
            (pts_2d[:, 0] - cx) / fx,
            (pts_2d[:, 1] - cy) / fy,
        ]
    )
    object_points = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float32)]
    ).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    image_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        video.camera_matrix,
        video.distortion_coefficients,
    )
    distorted = image_points.reshape(-1, 2)
    if pts.ndim == 1:
        return distorted[0]
    return distorted


def get_undistort_valid_fraction(video: Any) -> float:
    """Return fraction of pixels that remain valid after undistortion."""
    cache = getattr(video, "_undistort_valid_fraction", None)
    if cache is None:
        map1, map2, _ = get_undistort_maps(video)
        mask = np.ones((video.height, video.width), dtype=np.uint8)
        undistorted = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
        valid_fraction = float(np.count_nonzero(undistorted) / undistorted.size)
        setattr(video, "_undistort_valid_fraction", valid_fraction)
        return valid_fraction
    return float(cache)


def resolve_processing_window(
    video: Any,
    processing_window: Optional[tuple[int | float, int | float]],
    processing_window_unit: Literal["frame", "time", "timestamp"],
) -> tuple[int, int]:
    """Resolve a processing window to inclusive frame indices."""
    if processing_window is None:
        return 0, len(video.ts) - 1

    start, end = processing_window
    if processing_window_unit == "frame":
        return int(start), int(end)
    if processing_window_unit == "time":
        start_idx = int(np.searchsorted(video.times, start, side="left"))
        end_idx = int(np.searchsorted(video.times, end, side="right")) - 1
        return start_idx, end_idx
    if processing_window_unit == "timestamp":
        start_idx = int(np.searchsorted(video.ts, start, side="left"))
        end_idx = int(np.searchsorted(video.ts, end, side="right")) - 1
        return start_idx, end_idx

    raise ValueError(
        "processing_window_unit must be one of 'frame', 'time', or 'timestamp'."
    )
