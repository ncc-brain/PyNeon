import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video import SceneVideo


def detect_aruco(
    video: "SceneVideo",
    dictionary: str = "DICT_4X4_50",
    skip_frames: int = 1,
    adaptive_thresh: bool = True,
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Detect ArUco markers in a video and report their corner coordinates per frame.

    Parameters
    ----------
    video : SceneVideo
        Scene video to detect ArUco markers from.
    dictionary : str, optional
        Name of the ArUco dictionary to use (default: 'DICT_4X4_50').
        Options include:
            'DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250', 'DICT_7X7_1000', etc.
    skip_frames : int, optional
        Process every Nth frame (default 1).
    adaptive_thresh : bool, optional
        Use adaptive thresholding for better contrast in varying lighting (default True).
    return_diagnostics : bool, optional
        If True, include diagnostic information like detection confidence.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'timestamp [ns]' (index)
        - 'processed_frame_idx'
        - 'frame_idx'
        - 'tag_id'
        - 'corners' (4x2 ndarray)
        - 'center' (1x2 ndarray)
        - 'method' = 'aruco'
    """

    # -------------------------------------------------------------------
    # 1. Load ArUco dictionary and parameters
    # -------------------------------------------------------------------
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "OpenCV ArUco module not found. Install via `pip install opencv-contrib-python`."
        )

    # List of all available dictionaries
    available_dicts = {
        k: v for k, v in cv2.aruco.__dict__.items() if k.startswith("DICT_")
    }
    if dictionary not in available_dicts:
        raise ValueError(
            f"Unknown dictionary '{dictionary}'. Available: {list(available_dicts.keys())}"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(available_dicts[dictionary])
    parameters = cv2.aruco.DetectorParameters()

    if adaptive_thresh:
        parameters.adaptiveThreshConstant = 7

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # -------------------------------------------------------------------
    # 2. Iterate through frames
    # -------------------------------------------------------------------
    total_frames = len(video.ts)
    detections = []
    processed_frame_idx = 0

    for frame_idx in tqdm(
        range(0, total_frames, skip_frames), desc="Detecting ArUco markers"
    ):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            continue

        for marker_corners, marker_id in zip(corners_list, ids.flatten()):
            corners = marker_corners.reshape(4, 2)
            center = np.mean(corners, axis=0)

            result = {
                "processed_frame_idx": processed_frame_idx,
                "frame_idx": frame_idx,
                "timestamp [ns]": video.ts[frame_idx],
                "tag_id": int(marker_id),
                "corners": corners.astype(np.float32),
                "center": center.astype(np.float32),
            }

            if return_diagnostics:
                result["num_rejected"] = len(rejected)

            detections.append(result)

        processed_frame_idx += 1

    # -------------------------------------------------------------------
    # 3. Build DataFrame
    # -------------------------------------------------------------------
    df = pd.DataFrame(detections)
    if df.empty:
        return df

    df["method"] = "aruco"
    df.set_index("timestamp [ns]", inplace=True)
    return df
