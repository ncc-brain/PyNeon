from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .constants import DETECTION_COLUMNS, MARKERS_LAYOUT_COLUMNS

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

def _get_id(detection: pd.Series) -> Optional[str]:
    if "marker name" in detection:
        return str(detection["marker name"])
    elif "surface id" in detection:
        return str(detection["surface id"])
    return None

def _validate_marker_layout(marker_layout: pd.DataFrame) -> pd.DataFrame:
    """
    Validate marker layout dataframe and prepare corners.
    Expects columns: "marker name", "center x", "center y", "size".
    """
    required_columns = ["marker name", "center x", "center y", "size"]
    for col in required_columns:
        if col not in marker_layout.columns:
            raise ValueError(f"Marker layout must contain '{col}' column.")
    
    return _prepare_marker_layout(marker_layout)

def _validate_surface_layout(surface_layout: np.ndarray) -> np.ndarray:
    """
    Validate surface layout corners array.
    Expects a 4x2 array representing the four corners of the surface.
    """
    if not isinstance(surface_layout, np.ndarray):
        raise TypeError("Surface layout must be a numpy array.")
    if surface_layout.shape != (4, 2):
        raise ValueError(f"Surface layout must have shape (4, 2), got {surface_layout.shape}.")
    
    return surface_layout.astype(np.float32)

def _prepare_marker_layout(marker_layout: pd.DataFrame) -> pd.DataFrame:
    """Prepare marker layout by computing corner coordinates from center and size."""
    marker_layout = marker_layout.copy()
    marker_layout["corners"] = marker_layout.apply(
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
    return marker_layout

def _find_reference_lookup(layout: pd.DataFrame, id_col: str) -> dict:
    """Create a lookup dictionary mapping IDs to corner coordinates."""
    if id_col not in layout.columns:
        raise ValueError(f"Layout must contain '{id_col}' column for reference lookup.")
    
    lookup = {}
    for _, row in layout.iterrows():
        marker_id = row[id_col]
        if marker_id in lookup:
            raise ValueError(f"Duplicate {id_col} {marker_id} found in layout.")
        lookup[marker_id] = row["corners"]
    
    return lookup

def _compute_marker_homography(
    detections: pd.DataFrame,
    marker_layout: pd.DataFrame,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> pd.DataFrame:
    """
    Compute homographies for marker-based detections.
    
    Parameters
    ----------
    detections : pd.DataFrame
        DataFrame containing per-frame detections with corner coordinates and marker names.
    marker_layout : pd.DataFrame
        Marker layout with columns: "marker name", "center x", "center y", "size".
    method, ransacReprojThreshold, maxIters, confidence
        Parameters for cv2.findHomography.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with homography matrices indexed by timestamp.
    """
    # Validate and prepare marker layout
    prepared_layout = _validate_marker_layout(marker_layout)
    marker_lookup = _find_reference_lookup(prepared_layout, id_col="marker name")
    
    homography_for_frame = {}
    unique_timestamps = detections.index.unique()

    for ts in tqdm(unique_timestamps, desc="Computing marker homographies"):
        frame_detections = detections.loc[ts]

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

            # Get the marker name from the detection
            marker_id = _get_id(detection)
            if marker_id is None or marker_id not in marker_lookup:
                continue

            ref_corners = marker_lookup[marker_id]

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

    return pd.DataFrame.from_records(records)

def _compute_surface_homography(
    detections: pd.DataFrame,
    surface_layout: np.ndarray,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> pd.DataFrame:
    """
    Compute homographies for surface-based detections.
    
    Parameters
    ----------
    detections : pd.DataFrame
        DataFrame containing per-frame detections with corner coordinates.
    surface_layout : np.ndarray
        Surface corner coordinates as a 4x2 array.
    method, ransacReprojThreshold, maxIters, confidence
        Parameters for cv2.findHomography.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with homography matrices indexed by timestamp.
    """
    # Validate surface layout
    surface_corners = _validate_surface_layout(surface_layout)
    
    homography_for_frame = {}
    unique_timestamps = detections.index.unique()

    for ts in tqdm(unique_timestamps, desc="Computing surface homographies"):
        frame_detections = detections.loc[ts]

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

            world_points.extend(corners_detected)
            surface_points.extend(surface_corners)

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

    return pd.DataFrame.from_records(records)


@fill_doc
def find_homographies(
    detections: Stream | pd.DataFrame,
    marker_layout: pd.DataFrame = None,
    surface_layout: np.ndarray = None,
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

    For marker detections, detections should also contain a "marker name" column.
    For surface detections, provide surface_layout as a 4x2 numpy array.

    Parameters
    ----------
    detections : Stream or pandas.DataFrame
        Stream or DataFrame containing per-frame detections with corner coordinates.
    marker_layout : pd.DataFrame, optional
        Marker layout with columns: "marker name", "center x", "center y", "size".
    surface_layout : np.ndarray, optional
        Surface corner coordinates as a 4x2 array.
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

    if marker_layout is None and surface_layout is None:
        raise ValueError("Either marker_layout or surface_layout must be provided.")

    if marker_layout is not None and surface_layout is not None:
        raise ValueError("Only one of marker_layout or surface_layout should be provided.")

    # Route to appropriate helper function
    if marker_layout is not None:
        result_df = _compute_marker_homography(
            detection_df,
            marker_layout,
            method=method,
            ransacReprojThreshold=ransacReprojThreshold,
            maxIters=maxIters,
            confidence=confidence,
        )
    else:
        result_df = _compute_surface_homography(
            detection_df,
            surface_layout,
            method=method,
            ransacReprojThreshold=ransacReprojThreshold,
            maxIters=maxIters,
            confidence=confidence,
        )

    if result_df.empty:
        raise ValueError("No homographies could be computed from the detections.")

    return Stream(result_df)


