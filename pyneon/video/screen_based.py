import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video import SceneVideo


def detect_screen_corners(
    video: "SceneVideo",
    skip_frames: int = 1,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.9,
    brightness_threshold: int = 180,
    return_debug: bool = False,
) -> pd.DataFrame:
    """
    Detect rectangular bright regions (e.g., screens) in video frames and extract
    their corner coordinates.

    Parameters
    ----------
    video : SceneVideo
        Scene video to detect screen corners from.
    skip_frames : int, optional
        Process every Nth frame (default 1 = every frame).
    min_area_ratio : float, optional
        Minimum area ratio of detected contour relative to frame size (default 0.05).
    max_area_ratio : float, optional
        Maximum area ratio of detected contour relative to frame size (default 0.9).
    brightness_threshold : int, optional
        Threshold (0–255) for binarizing the grayscale image. Default 180.
    return_debug : bool, optional
        If True, include diagnostic information such as contour area and hierarchy.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'timestamp [ns]' (index)
        - 'processed_frame_idx'
        - 'frame_idx'
        - 'tag_id' (always 0 for single rectangle)
        - 'corners' : ndarray (4,2) coordinates (TL, TR, BR, BL)
        - 'center' : ndarray (1,2) center of rectangle
    """

    total_frames = len(video.ts)
    detections = []
    processed_frame_idx = 0

    frames_to_process = range(0, total_frames, skip_frames)
    for actual_frame_idx in tqdm(frames_to_process, desc="Detecting screen corners"):
        video.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        frame_area = w * h

        # Threshold bright areas
        _, thresh = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        best_contour = None
        best_area = 0

        # Select the largest plausible rectangular contour
        for c in contours:
            area = cv2.contourArea(c)
            area_ratio = area / frame_area
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue

            # Approximate polygon to check rectangular shape
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and area > best_area:
                best_contour = approx
                best_area = area

        if best_contour is None:
            continue

        # Order corners: TL, TR, BR, BL
        corners = best_contour.reshape(-1, 2).astype(np.float32)
        corners = order_corners(corners)

        center = np.mean(corners, axis=0)

        result = {
            "processed_frame_idx": processed_frame_idx,
            "frame_idx": actual_frame_idx,
            "timestamp [ns]": video.ts[actual_frame_idx],
            "tag_id": 0,
            "corners": corners,
            "center": center,
        }

        if return_debug:
            result["area"] = best_area
            result["contour"] = best_contour

        detections.append(result)
        processed_frame_idx += 1

    df = pd.DataFrame(detections)
    if df.empty:
        return df

    df["method"] = "screen"
    df.set_index("timestamp [ns]", inplace=True)
    return df


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Sort 4 corner points into TL, TR, BR, BL order.
    """
    if corners.shape != (4, 2):
        raise ValueError("Corners must be a (4, 2) array")

    # Compute centroid
    center = np.mean(corners, axis=0)

    # Sort by angle around centroid
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sort_idx = np.argsort(angles)
    corners = corners[sort_idx]

    # Ensure TL is first (top-left = smallest sum of coordinates)
    s = corners.sum(axis=1)
    min_idx = np.argmin(s)
    corners = np.roll(corners, -min_idx, axis=0)

    # Ensure order is TL, TR, BR, BL (clockwise)
    if np.cross(corners[1] - corners[0], corners[2] - corners[0]) < 0:
        corners = np.flip(corners, axis=0)

    return corners
