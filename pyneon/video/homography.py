import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .utils import _validate_marker_layout, _validate_surface_layout

def _reshape_corners(detection: pd.Series) -> np.ndarray:
    return np.array(
        [
            [detection["top left x [px]"], detection["top left y [px]"]],
            [detection["top right x [px]"], detection["top right y [px]"]],
            [detection["bottom right x [px]"], detection["bottom right y [px]"]],
            [detection["bottom left x [px]"], detection["bottom left y [px]"]],
        ],
        dtype=np.float32,
    )

@fill_doc
def find_homographies(
    detections: Stream,
    layout: pd.DataFrame | np.ndarray,
    min_markers: int = 2,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> Stream:
    """
    Compute a homography (3x3 matrix) for each frame using marker
    or surface-corner detections.

    Parameters
    ----------
    detections : Stream
        Stream containing per-frame detections with corner coordinates.
        Obtained from :meth:`Video.detect_markers` or :meth:`Video.detect_surfaces`.
    layout : pd.DataFrame or np.ndarray
        If using marker detections, provide a DataFrame with columns:
    surface_layout : np.ndarray, optional
        Surface corner coordinates as a 4x2 array.
    %(find_homographies_params)s

    Returns
    -------
    %(find_homographies_returns)s
    """
    detection_df = detections.data
    is_marker_detection = isinstance(layout, pd.DataFrame)

    # Route to appropriate helper function
    if is_marker_detection:
        # Validate marker layout
        _validate_marker_layout(layout)
        # Compute corner coordinates for each marker in the layout
        layout = layout.copy()
        layout["corners"] = layout.apply(
            lambda row: np.array(
                [
                    [
                        row["center x"] - row["size"] / 2,
                        row["center y"] - row["size"] / 2,
                    ],
                    [
                        row["center x"] + row["size"] / 2,
                        row["center y"] - row["size"] / 2,
                    ],
                    [
                        row["center x"] + row["size"] / 2,
                        row["center y"] + row["size"] / 2,
                    ],
                    [
                        row["center x"] - row["size"] / 2,
                        row["center y"] + row["size"] / 2,
                    ],
                ],
                dtype=np.float32,
            ),
            axis=1,
        )
        # Construct a lookup dictionary with marker name being key and corners being value
        surface_pts_lookup = {row["marker name"]: row["corners"] for _, row in layout.iterrows()}
    else:
        _validate_surface_layout(layout)
        surface_pts_lookup = {"surface_0": layout}

    homography_per_frame = {}
    unique_timestamps = detection_df.index.unique()

    for ts in tqdm(unique_timestamps, desc="Computing surface homographies"):
        frame_detections = detection_df.loc[ts]
        if isinstance(frame_detections, pd.Series):
            frame_detections = frame_detections.to_frame().T
        
        if is_marker_detection and len(frame_detections) < min_markers:
            continue

        camera_pts_all = []
        surface_pts_all = []

        for _, detection in frame_detections.iterrows():
            camera_pts = _reshape_corners(detection)
            name = detection["marker name"] if is_marker_detection else "surface_0"
            surface_pts = surface_pts_lookup[name]
            if camera_pts.shape != (4, 2):
                raise ValueError(
                    f"Detected corners must have shape (4, 2), got {camera_pts.shape}"
                )

            camera_pts_all.extend(camera_pts)
            surface_pts_all.extend(surface_pts)

        camera_pts_all = np.array(camera_pts_all, dtype=np.float32).reshape(-1, 2)
        surface_pts_all = np.array(surface_pts_all, dtype=np.float32).reshape(-1, 2)

        homography, _ = cv2.findHomography(
            camera_pts_all,
            surface_pts_all,
            method=method,
            ransacReprojThreshold=ransacReprojThreshold,
            maxIters=maxIters,
            confidence=confidence,
        )
        homography_per_frame[ts] = homography

    records = []
    for ts, homography in homography_per_frame.items():
        record = {"timestamp [ns]": ts}
        if homography is not None:
            for i in range(3):
                for j in range(3):
                    record[f"homography ({i},{j})"] = homography[i, j]
            records.append(record)

    if not records:
        raise ValueError("No homographies could be computed from the detections.")

    homographies_df = pd.DataFrame(records)
    homographies_df.set_index("timestamp [ns]", inplace=True)
    return Stream(homographies_df)
