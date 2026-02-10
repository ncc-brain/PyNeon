from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream

if TYPE_CHECKING:
    from .video import Video

def detect_surface(
    video: "Video",
    skip_frames: int = 1,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.98,
    brightness_threshold: int = 180,
    adaptive: bool = True,
    morph_kernel: int = 5,
    decimate: float = 1.0,
    mode: Literal["largest", "best", "all"] = "largest",
) -> Stream:
    """
    Detect bright rectangular regions (e.g., projected screens or monitors)
    in video frames using luminance-based contour detection.

    This function identifies one or more rectangular contours per frame based on
    their brightness and geometry. It supports multiple selection modes:
    returning either all rectangular candidates, only the largest, or the most
    rectangular one according to a geometric certainty score.

    The resulting DataFrame can be used directly in homography estimation
    pipelines to map projected screen coordinates onto the camera view.

    Parameters
    ----------
    video : Video
        Scene video object supporting OpenCV-like `set()` and `read()` methods,
        and providing frame timestamps (`video.ts`).
    skip_frames : int, optional
        Process every Nth frame (default 1 = process all frames).
    min_area_ratio : float, optional
        Minimum contour area relative to frame area. Contours smaller than this
        ratio are ignored. Default is 0.01 (1% of frame area).
    max_area_ratio : float, optional
        Maximum contour area relative to frame area. Contours larger than this
        ratio are ignored. Default is 0.98.
    brightness_threshold : int, optional
        Fixed threshold for binarization when `adaptive=False`. Default is 180.
    adaptive : bool, optional
        If True (default), use adaptive thresholding to handle varying
        illumination across frames.
    morph_kernel : int, optional
        Kernel size for morphological closing (default 5). Use 0 to disable
        morphological operations.
    decimate : float, optional
        Downsampling factor for faster processing (e.g., 0.5 halves resolution).
        Detected coordinates are automatically rescaled back. Default is 1.0.
    mode : {"largest", "best", "all"}, optional
        Selection mode determining which contours to return per frame:

        - "largest" : Return only the largest valid rectangular contour.
          Useful when the screen is the outermost bright region. (Default)
        - "best" : Return the contour that most closely resembles a
          perfect rectangle (lowest corner-angle variance and balanced
          aspect ratio).
        - "all" : Return all valid rectangular contours (outer and inner
          overlapping rectangles). Useful when both screen and inner
          projected content need to be distinguished.

    Returns
    -------
    Stream

    One row per detected rectangular contour with columns:
        - "processed_frame_idx" : int
        - "processed frame index" : int
        - "frame_idx"           : int
        - "frame index"         : int
        - "timestamp [ns]"      : int64
        - "tag_id"              : int (sequential ID per contour in frame)
        - "corners"             : ndarray (4, 2) corner coordinates
        - "center"              : ndarray (1, 2) center point
        - "method"              : str ("screen")
        - "area_ratio"          : float
        - "score"               : float
    """

    if skip_frames < 1:
        raise ValueError("skip_frames must be >= 1")
    if decimate <= 0:
        raise ValueError("decimate must be > 0")

    total_frames = len(video.ts)
    detections = []
    processed_frame_idx = 0
    frames_to_process = range(0, total_frames, skip_frames)

    columns = [
        "timestamp [ns]",
        "processed_frame_idx",
        "processed frame index",
        "frame_idx",
        "frame index",
        "tag_id",
        "corners",
        "center",
        "method",
        "area_ratio",
        "score",
    ]

    for actual_frame_idx in tqdm(frames_to_process, desc="Detecting screen corners"):
        gray = video.read_gray_frame_at(actual_frame_idx)
        if gray is None:
            break

        if decimate != 1.0:
            gray = cv2.resize(gray, None, fx=decimate, fy=decimate)

        h, w = gray.shape[:2]
        frame_area = w * h

        if adaptive:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -10
            )
        else:
            _, thresh = cv2.threshold(
                gray, brightness_threshold, 255, cv2.THRESH_BINARY
            )

        if morph_kernel > 0:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            processed_frame_idx += 1
            continue

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            area_ratio = area / frame_area
            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            corners = approx.reshape(-1, 2).astype(np.float32)
            corners = _order_corners_tl_tr_br_bl(corners)
            if decimate != 1.0:
                corners /= decimate

            score = _rectangular_score(corners)
            candidates.append(
                {
                    "corners": corners,
                    "area_ratio": area_ratio,
                    "score": score,
                }
            )

        if not candidates:
            processed_frame_idx += 1
            continue

        if mode == "largest":
            selected = [max(candidates, key=lambda x: x["area_ratio"])]
        elif mode == "best":
            selected = [max(candidates, key=lambda x: x["score"])]
        elif mode == "all":
            selected = candidates
        else:
            raise ValueError(
                f"Unknown mode '{mode}', must be 'largest', 'best', or 'all'."
            )

        for cid, sel in enumerate(selected):
            corners = sel["corners"]
            center = np.mean(corners, axis=0)
            detections.append(
                {
                    "processed_frame_idx": processed_frame_idx,
                    "processed frame index": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
                    "frame index": actual_frame_idx,
                    "timestamp [ns]": int(video.ts[actual_frame_idx]),
                    "tag_id": cid,
                    "corners": corners,
                    "center": center,
                    "method": "screen",
                    "area_ratio": sel["area_ratio"],
                    "score": sel["score"],
                }
            )

        processed_frame_idx += 1

    df = pd.DataFrame(detections)
    if df.empty:
        print("Warning: No screen contours detected.")
        df = pd.DataFrame(columns=columns)

    df.set_index("timestamp [ns]", inplace=True)
    return Stream(df)


def _rectangular_score(pts: np.ndarray) -> float:
    v = np.diff(np.vstack([pts, pts[0]]), axis=0)
    lengths = np.linalg.norm(v, axis=1)
    angles = []
    for i in range(4):
        v1 = v[i - 1] / lengths[i - 1]
        v2 = v[i] / lengths[i]
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        angles.append(ang)
    angle_var = np.var(angles)
    aspect_ratio = max(lengths) / min(lengths)
    # Ensure a plain Python float for typeguard.
    return float(-angle_var - abs(aspect_ratio - 1) * 10)


def _order_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    if corners.shape != (4, 2):
        raise ValueError("Expected (4,2) corners array.")

    sorted_by_y = corners[np.argsort(corners[:, 1])]

    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype=np.float32)
