from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .constants import DETECTION_COLUMNS
from .utils import (
    _verify_format,
    distort_points,
    get_undistort_valid_fraction,
    resolve_processing_window,
)

if TYPE_CHECKING:
    from .video import Video


@fill_doc
def detect_surface(
    video: "Video",
    step: int = 1,
    processing_window: tuple[int | float, int | float] | None = None,
    processing_window_unit: Literal["frame", "time", "timestamp"] = "frame",
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.98,
    brightness_threshold: int = 180,
    adaptive: bool = True,
    morph_kernel: int = 5,
    decimate: float = 1.0,
    mode: Literal["largest", "best"] = "largest",
    report_diagnostics: bool = False,
    undistort: bool = False,
) -> Stream:
    """
    Detect bright rectangular regions (e.g., projected surfaces or monitors)
    in video frames using luminance-based contour detection.

    This function identifies one or more rectangular contours per frame based on
    their brightness and geometry. It supports multiple selection modes:
    returning either all rectangular candidates, only the largest, or the most
    rectangular one according to a geometric certainty score.

    The resulting DataFrame can be used directly in homography estimation
    pipelines to map projected surface coordinates onto the camera view.

    Parameters
    ----------
    video : Video
        Video instance supporting OpenCV-like `set()` and `read()` methods,
        and providing frame timestamps (`video.ts`).
    %(detect_surface_params)s

    Returns
    -------
    %(detect_surface_return)s
    """

    if step < 1:
        raise ValueError("step must be >= 1")
    if decimate <= 0:
        raise ValueError("decimate must be > 0")

    start_frame_idx, end_frame_idx = resolve_processing_window(
        video,
        processing_window,
        processing_window_unit,
    )
    detections = []
    frames_to_process = range(start_frame_idx, end_frame_idx + 1, step)

    # Ensure video is at the beginning before processing
    video.reset()

    valid_fraction = None
    if undistort:
        valid_fraction = get_undistort_valid_fraction(video)

    for actual_frame_idx in tqdm(frames_to_process, desc="Detecting surface corners"):
        frame = video.read_frame_at(actual_frame_idx)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_frame is None:
            break

        if undistort:
            gray_frame = video.undistort_frame(gray_frame)

        if decimate != 1.0:
            gray_frame = cv2.resize(gray_frame, None, fx=decimate, fy=decimate)

        h, w = gray_frame.shape[:2]
        frame_area = w * h

        if adaptive:
            thresh = cv2.adaptiveThreshold(
                gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -10
            )
        else:
            _, thresh = cv2.threshold(
                gray_frame, brightness_threshold, 255, cv2.THRESH_BINARY
            )

        if morph_kernel > 0:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            area_ratio = area / frame_area
            if undistort and valid_fraction:
                area_ratio /= valid_fraction
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
            continue

        if mode == "largest":
            selected = max(candidates, key=lambda x: x["area_ratio"])
        elif mode == "best":
            selected = max(candidates, key=lambda x: x["score"])
        else:
            raise ValueError(
                f"Unknown mode '{mode}', must be 'largest' or 'best'."
            )

        corners = selected["corners"]
        center = np.mean(corners, axis=0)
        if undistort:
            corners = distort_points(video, corners)
            center = distort_points(video, center)
        detection_row = {
            "frame index": actual_frame_idx,
            "timestamp [ns]": int(video.ts[actual_frame_idx]),
            "surface id": 0,
            "top left x [px]": corners[0, 0],
            "top left y [px]": corners[0, 1],
            "top right x [px]": corners[1, 0],
            "top right y [px]": corners[1, 1],
            "bottom right x [px]": corners[2, 0],
            "bottom right y [px]": corners[2, 1],
            "bottom left x [px]": corners[3, 0],
            "bottom left y [px]": corners[3, 1],
            "center x [px]": center[0],
            "center y [px]": center[1],
        }
        if report_diagnostics:
            detection_row["area_ratio"] = selected["area_ratio"]
            detection_row["score"] = selected["score"]
        detections.append(detection_row)

    df = pd.DataFrame(detections)
    if df.empty:
        raise ValueError("No surfaces detected in the specified processing window.")


    df.set_index("timestamp [ns]", inplace=True)
    _verify_format(df, DETECTION_COLUMNS)
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
