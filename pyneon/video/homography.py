from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .constants import DETECTION_COLUMNS, MARKERS_LAYOUT_COLUMNS
from .layout import Layout

def _extract_corners(detection: pd.Series) -> np.ndarray:
    return np.array(
        [
            [detection["top left x [px]"], detection["top left y [px]"]],
            [detection["top right x [px]"], detection["top right y [px]"]],
            [detection["bottom right x [px]"], detection["bottom right y [px]"]],
            [detection["bottom left x [px]"], detection["bottom left y [px]"]],
        ],
        dtype=np.float32,
    )

def _get_id(detection: pd.Series) -> Optional[int]:
    if "marker id" in detection:
        return int(detection["marker id"])
    elif "surface id" in detection:
        return int(detection["surface id"])
    return None

@fill_doc
def find_homographies(
    detections: Stream | pd.DataFrame,
    layout: Layout,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> Stream:
    """
    Compute a homography for each frame using marker or surface-corner detections.

    Detections should contain corner coordinates with columns:
    "top left x [px]", "top left y [px]", "top right x [px]", "top right y [px]",
    "bottom right x [px]", "bottom right y [px]", "bottom left x [px]", "bottom left y [px]".

    For marker detections, detections should also contain a "marker id" column.
    For surface detections, detections should also contain a "surface id" column (optional if using a single surface).

    Parameters
    ----------
    detections : Stream or pandas.DataFrame
        Stream or DataFrame containing per-frame detections with corner coordinates and ID columns.
    layout : Layout
        Layout object containing reference corners for markers or surfaces.
    %(find_homographies_params)s

    Returns
    -------
    %(find_homographies_return)s
    """
    if isinstance(detections, Stream):
        detection_df = detections.data
    else:
        detection_df = detections

    if detection_df.empty:
        raise ValueError("Detections are empty.")

    if layout is None:
        raise ValueError("Layout must be provided to compute homographies from detections.")

    if layout.marker_lookup is None or len(layout.marker_lookup) == 0:
        raise ValueError("Layout must contain marker/surface references.")

    detection_mode = layout.source
    homography_for_frame = {}
    unique_timestamps = detection_df.index.unique()

    for ts in tqdm(unique_timestamps, desc="Computing homographies"):
        frame_detections = detection_df.loc[ts]

        if isinstance(frame_detections, pd.Series):
            frame_detections = frame_detections.to_frame().T

        world_points = []
        surface_points = []

        for _, detection in frame_detections.iterrows():
            corners_detected = _extract_corners(detection)

            if corners_detected.shape != (4, 2):
                raise ValueError(
                    f"Detected corners must have shape (4, 2), got {corners_detected.shape}"
                )

            # Get the ID from the detection (marker id or surface id)
            marker_id = _get_id(detection)
            if marker_id is None or marker_id not in layout.marker_lookup:
                continue

            ref_corners = layout.marker_lookup[marker_id]

            world_points.extend(corners_detected)
            surface_points.extend(ref_corners)

        if len(world_points) < 4:
            # Need at least 4 points to compute homography
            continue

        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
        surface_points = np.array(surface_points, dtype=np.float32).reshape(-1, 2)

        homography, _ = cv2.findHomography(
            world_points,
            surface_points,
            method=method,
            ransacReprojThreshold=ransacReprojThreshold,
            maxIters=maxIters,
            confidence=confidence,
        )
        homography_for_frame[ts] = homography

    records = []
    for ts, homography in homography_for_frame.items():
        record = {"timestamp [ns]": ts}
        if homography is not None:
            for i in range(3):
                for j in range(3):
                    record[f"homography ({i},{j})"] = homography[i, j]
            records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No homographies could be computed from the detections.")

    return Stream(df)


