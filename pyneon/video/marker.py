from typing import TYPE_CHECKING, Literal, Optional

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
    marker_family_to_dict,
    resolve_detection_window,
)

if TYPE_CHECKING:
    from .video import Video


@fill_doc
def detect_markers(
    video: "Video",
    marker_family: str | list[str],
    step: int = 1,
    detection_window: Optional[tuple[int | float, int | float]] = None,
    detection_window_unit: Literal["frame", "time", "timestamp"] = "frame",
    detector_parameters: Optional[cv2.aruco.DetectorParameters] = None,
    undistort: bool = False,
) -> Stream:
    """
    Detect fiducial markers (AprilTag or ArUco) in a video and report their data for every processed frame.

    Parameters
    ----------
    video : Video
        Scene video to detect markers from.
    %(detect_markers_params)s
    %(detect_markers_return)s
    """
    # Normalize marker family input to a list and create detectors for each
    families: list[str] = (
        marker_family if isinstance(marker_family, list) else [marker_family]
    )

    # Use provided detector_parameters or create a default instance
    if detector_parameters is None:
        detector_parameters = cv2.aruco.DetectorParameters()

    detectors: list[tuple[str, str, cv2.aruco.ArucoDetector]] = []
    for fam in families:
        fam_type, aruco_dict = marker_family_to_dict(fam)
        detectors.append(
            (fam, fam_type, cv2.aruco.ArucoDetector(aruco_dict, detector_parameters))
        )

    if step < 1:
        raise ValueError("step must be >= 1")

    start_frame_idx, end_frame_idx = resolve_detection_window(
        video,
        detection_window,
        detection_window_unit,
    )

    def _process_frame(frame_idx: int, gray_frame: np.ndarray) -> list[dict]:
        """Run detection on a single grayscale frame across all detectors."""
        records: list[dict] = []
        for fam_name, fam_type, det in detectors:
            all_corners, all_ids, _ = det.detectMarkers(gray_frame)
            if all_ids is None:
                continue

            for corners, marker_id in zip(all_corners, all_ids):
                corners = corners.reshape((4, 2))
                if fam_type == "april":
                    # For AprilTags, corners start with bottom right
                    # For ArUco, corners start with top left
                    # See https://stackoverflow.com/questions/79044142
                    corners = corners[[2, 3, 0, 1], :]
                center = np.mean(corners, axis=0)
                if undistort:
                    corners = distort_points(video, corners)
                    center = distort_points(video, center)
                records.append(
                    {
                        "timestamp [ns]": video.ts[frame_idx],
                        "frame index": frame_idx,
                        "marker family": fam_name,
                        "marker id": int(marker_id[0]),
                        "marker name": f"{fam_name}_{marker_id[0]}",
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
                )
        return records

    detected_markers = []
    frames_to_process = list(range(start_frame_idx, end_frame_idx + 1, step))

    # Ensure video is at the beginning before processing
    video.reset()  

    for frame_index in tqdm(frames_to_process, desc="Detecting markers"):
        gray_frame = video.read_gray_frame_at(frame_index)
        if gray_frame is None:
            break
        if undistort:
            gray_frame = video.undistort_frame(gray_frame)
        records = _process_frame(frame_index, gray_frame)
        detected_markers.extend(records)

    df = pd.DataFrame(detected_markers)
    if df.empty:
        raise ValueError("No marker detected.")

    df.set_index("timestamp [ns]", inplace=True)
    _verify_format(df, DETECTION_COLUMNS)
    return Stream(df)
