from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from .constants import MARKERS_LAYOUT_COLUMNS


def find_homographies(
    detections: Stream | pd.DataFrame,
    layout: pd.DataFrame | np.ndarray,
    valid_markers: int = 2,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> Stream:
    """
    Compute a homography for each frame using marker or surface-corner detections.

    The function automatically determines the detection and layout types
    based on the input data:

    - **Marker detections** from :meth:`Video.detect_markers` require a marker layout
      (DataFrame with "marker name", "size", "center x", "center y" columns).
    - **Surface-corner detections** (column: "corners")
      require surface layout (DataFrame with "corners" column, or a 4x2 numpy array)

    Parameters
    ----------
    detections : Stream or pandas.DataFrame
        Stream or DataFrame containing per-frame detections.
    layout : pandas.DataFrame or numpy.ndarray
        Layout specification for computing homographies. Format depends on the
        detection type:

        - For marker detections: DataFrame with columns "marker name", "size",
          "center x", "center y" (marker reference coordinates)
        - For surface-corner detections: DataFrame with a "corners" column
          containing 4x2 arrays per row, or a single 4x2 numpy array. If a
          "surface id" column is present, it's used to match detections to layout rows.
    valid_markers : int, optional
        Minimum number of markers required to compute a homography. Defaults to 2.
    method : int, optional
        Method used to compute a homography matrix. The following methods are possible:

        - 0 - a regular method using all the points, i.e., the least squares method
        - ``cv2.RANSAC`` - RANSAC-based robust method
        - ``cv2.LMEDS`` - Least-Median robust method
        - ``cv2.RHO`` - PROSAC-based robust method

        Defaults to ``cv2.LMEDS``.
    ransacReprojThreshold : float, optional
        Maximum allowed reprojection error to treat a point pair as an inlier
        (used in the RANSAC and RHO methods only). Defaults to 3.0.
    maxIters : int, optional
        The maximum number of RANSAC iterations. Defaults to 2000.
    confidence : float, optional
        Confidence level, between 0 and 1, for the estimated homography.
        Defaults to 0.995.

    Returns
    -------
    Stream
        A Stream indexed by 'timestamp [ns]' with columns
        'homography (0,0)' through 'homography (2,2)': The 9 elements of the
        flattened 3x3 homography matrix.
        
    Notes
    -----
    The flattened homography matrix as columns, and the fact that
    homographies is a Stream, allows for interpolation.
    """
    if isinstance(detections, Stream):
        detection_df = detections.data
    else:
        detection_df = detections
    if detection_df.empty:
        raise ValueError("Detections are empty.")

    is_marker = all(
        col in detection_df.columns
        for col in [
            "marker family",
            "marker id",
            "marker name",
        ]
    )
    is_surface = "surface id" in detection_df.columns

    # Validate and prepare layout based on detection type
    if is_marker:
        if not all(col in layout.columns for col in MARKERS_LAYOUT_COLUMNS):
            raise ValueError(
                f"Marker layout DataFrame must contain columns: {MARKERS_LAYOUT_COLUMNS}"
            )
        layout = layout.copy()
        # Compute marker corners from center positions and size
        layout["marker_corners"] = layout.apply(
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
    elif is_surface:
        layout, layout_by_marker, base_corners = _prepare_corner_layout(layout)
    else:
        raise ValueError(
            "Invalid detections. Please use outputs from detect_markers or detect_surface, or ensure the DataFrame has the required columns."
        )

    unique_timestamps = detection_df.index.unique()
    homography_for_frame = {}

    for ts in tqdm(unique_timestamps, desc="Computing homographies"):
        frame_detections = detection_df.loc[ts]

        if isinstance(frame_detections, pd.Series):
            frame_detections = frame_detections.to_frame().T
        if frame_detections.shape[0] < valid_markers:
            continue

        world_points = []
        surface_points = []

        if is_marker:
            for _, detection in frame_detections.iterrows():
                marker_name = detection["marker name"]
                if marker_name not in layout["marker name"].values:
                    continue

                world_corners = np.array(
                    [
                        [detection["top left x [px]"], detection["top left y [px]"]],
                        [detection["top right x [px]"], detection["top right y [px]"]],
                        [
                            detection["bottom right x [px]"],
                            detection["bottom right y [px]"],
                        ],
                        [
                            detection["bottom left x [px]"],
                            detection["bottom left y [px]"],
                        ],
                    ],
                    dtype=np.float32,
                )
                ref_corners = layout.loc[
                    layout["marker name"] == marker_name, "marker_corners"
                ].values[0]

                if world_corners.shape != (4, 2) or ref_corners.shape != (4, 2):
                    raise ValueError(
                        "Marker corners must have shape (4, 2), got "
                        f"{world_corners.shape} and {ref_corners.shape}"
                    )

                world_points.extend(world_corners)
                surface_points.extend(ref_corners)
        else:  # surface mode
            for _, detection in frame_detections.iterrows():
                world_corners = np.asarray(detection["corners"], dtype=np.float32)

                if world_corners.shape != (4, 2):
                    raise ValueError(
                        f"Detected corners must have shape (4, 2), got {world_corners.shape}"
                    )

                if layout_by_marker is not None:
                    marker_id = _get_id(detection)
                    if marker_id is None or marker_id not in layout_by_marker:
                        continue
                    ref_corners = layout_by_marker[marker_id]
                else:
                    ref_corners = base_corners

                world_points.extend(world_corners)
                surface_points.extend(ref_corners)

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


def _get_id(detection: pd.Series) -> Optional[int]:
    if "marker id" in detection:
        return int(detection["marker id"])
    return None


def _prepare_corner_layout(
    surface_layout: pd.DataFrame | np.ndarray,
) -> tuple[
    Optional[pd.DataFrame], Optional[dict[int, np.ndarray]], Optional[np.ndarray]
]:
    if isinstance(surface_layout, np.ndarray):
        if surface_layout.shape != (4, 2):
            raise ValueError("Layout array must have shape (4, 2).")
        base_corners = np.asarray(surface_layout, dtype=np.float32)
        return None, None, base_corners

    if "surface id" in surface_layout.columns:
        layout_by_marker = {}
        for _, row in surface_layout.iterrows():
            marker_id = int(row["surface id"])
            corners = np.asarray(row["corners"], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError("Each layout 'corners' entry must be shape (4, 2).")
            layout_by_marker[marker_id] = corners
        return surface_layout, layout_by_marker, None

    if len(surface_layout) != 1:
        raise ValueError(
            "surface_layout must have one row or include a 'surface id' column when using corner detections."
        )

    base_corners = np.asarray(surface_layout["corners"].iloc[0], dtype=np.float32)
    if base_corners.shape != (4, 2):
        raise ValueError("surface_layout 'corners' must have shape (4, 2).")
    return surface_layout, None, base_corners
