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
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.98,
    brightness_threshold: int = 180,
    adaptive: bool = True,
    morph_kernel: int = 5,
    decimate: float = 1.0,
    mode: str = "largest",
) -> pd.DataFrame:
    """
    Detect bright rectangular regions (e.g., projected screens or monitors)
    in video frames using luminance-based contour detection.

    This function identifies one or more rectangular contours per frame based on
    their brightness and geometry. It supports multiple selection modes:
    returning either all rectangular candidates, only the largest, or the most
    "rectangular" one according to a geometric certainty score.

    The resulting DataFrame can be used directly in homography estimation
    pipelines to map projected screen coordinates onto the camera view.

    Parameters
    ----------
    video : SceneVideo
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

        - `"largest"` : Return only the largest valid rectangular contour.
            Useful when the screen is the outermost bright region. *(Default)*
        - `"best"` : Return the contour that most closely resembles a
            perfect rectangle (lowest corner-angle variance and balanced
            aspect ratio).
        - `"all"` : Return all valid rectangular contours (outer and inner
            overlapping rectangles). Useful when both screen and inner
            projected content need to be distinguished.

    Returns
    -------
    pandas.DataFrame

    One row per detected rectangular contour with columns:
        - "processed_frame_idx" : int
        - "frame_idx"           : int
        - "timestamp [ns]"      : int64
        - "tag_id"              : int (sequential ID per contour in frame)
        - "corners"             : ndarray (4, 2) corner coordinates
        - "center"              : ndarray (1, 2) center point
        - "method"              : str ("screen")
        - "area_ratio"          : float (if return_debug=True)
        - "score"               : float (if return_debug=True)

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

        # optional decimation
        if decimate != 1.0:
            frame_small = cv2.resize(frame, None, fx=decimate, fy=decimate)
        else:
            frame_small = frame

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        frame_area = w * h

        # --- Thresholding ---
        if adaptive:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -10
            )
        else:
            _, thresh = cv2.threshold(
                gray, brightness_threshold, 255, cv2.THRESH_BINARY
            )

        # --- Morphological cleanup ---
        if morph_kernel > 0:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # --- Find all contours, including nested ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
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

            # --- Certainty metric: how rectangular is it? ---
            def rectangular_score(pts):
                # compute side lengths and angles
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
                # smaller variance + aspect close to 1 → higher score
                return -angle_var - abs(aspect_ratio - 1) * 10

            score = rectangular_score(corners)
            candidates.append(
                {
                    "corners": corners,
                    "area_ratio": area_ratio,
                    "score": score,
                }
            )

        if not candidates:
            continue

        # --- choose subset depending on mode ---
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

        # --- record detections ---
        for cid, sel in enumerate(selected):
            corners = sel["corners"]
            center = np.mean(corners, axis=0)
            detections.append(
                {
                    "processed_frame_idx": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
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
        return df

    df.set_index("timestamp [ns]", inplace=True)
    return df


def _order_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """
    Order 4 corners in TL–TR–BR–BL order (clockwise).

    Parameters
    ----------
    corners : np.ndarray (4, 2)
        Unordered corner coordinates.

    Returns
    -------
    np.ndarray (4, 2)
        Ordered corners (TL, TR, BR, BL).
    """
    if corners.shape != (4, 2):
        raise ValueError("Expected (4,2) corners array.")

    # Sort by y (top to bottom)
    sorted_by_y = corners[np.argsort(corners[:, 1])]

    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]

    # Sort each by x (left to right)
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    # Combine
    tl, tr = top
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype=np.float32)
