from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .marker import DETECTED_MARKERS_COLUMNS, MARKERS_LAYOUT_COLUMNS


@fill_doc
def find_homographies(
    detected_markers: Stream | pd.DataFrame,
    marker_layout: Optional[pd.DataFrame] = None,
    surface_layout: Optional[pd.DataFrame | np.ndarray] = None,
    valid_markers: int = 2,
    method: int = cv2.LMEDS,
    ransacReprojThreshold: float = 3.0,
    maxIters: int = 2000,
    confidence: float = 0.995,
) -> Stream:
    """
    Compute a homography for each frame using marker or screen-corner detections.

    For marker detections, provide ``marker_layout`` with columns defined in
    ``%(marker_layout)s``. For screen-corner detections (from
    :func:`pyneon.video.detect_screen_corners`), provide ``surface_layout``
    containing a ``corners`` column with a 4x2 array per row, or a single 4x2
    numpy array. If ``surface_layout`` contains a ``tag_id`` column, it is used
    to match detections to layout rows.

    Parameters
    ----------
    detected_markers : Stream or pandas.DataFrame
        Stream or DataFrame containing per-frame detections.
    marker_layout : pandas.DataFrame, optional
        Marker layout DataFrame (for fiducials). Required for marker detections.
    surface_layout : pandas.DataFrame or numpy.ndarray, optional
        Screen layout with a ``corners`` column or a 4x2 numpy array. Required
        for screen-corner detections.
    %(find_homographies_params)s

    Returns
    -------
    %(find_homographies_return)s
    """
    if isinstance(detected_markers, Stream):
        detection_df = detected_markers.data
    else:
        detection_df = detected_markers

    if detection_df.empty:
        raise ValueError("Detections are empty.")

    uses_marker_columns = DETECTED_MARKERS_COLUMNS.issubset(detection_df.columns)
    uses_corner_column = "corners" in detection_df.columns

    if marker_layout is not None and surface_layout is not None:
        raise ValueError("Provide only one of marker_layout or surface_layout.")

    if uses_marker_columns:
        if marker_layout is None:
            raise ValueError("marker_layout is required for marker detections.")
        layout_df = _prepare_marker_layout(marker_layout)
        layout_by_tag = None
        base_corners = None
        detection_mode = "marker"
    elif uses_corner_column:
        if surface_layout is None:
            raise ValueError("surface_layout is required for corner detections.")
        layout_df, layout_by_tag, base_corners = _prepare_corner_layout(surface_layout)
        detection_mode = "corners"
    else:
        raise ValueError(
            "Detections must contain marker corner columns or a 'corners' column."
        )

    unique_timestamps = detection_df.index.unique()
    homography_for_frame = {}
    required_points = max(valid_markers, 1) * 4

    for ts in tqdm(unique_timestamps, desc="Computing homographies"):
        frame_detections = detection_df.loc[ts]

        if isinstance(frame_detections, pd.Series):
            frame_detections = frame_detections.to_frame().T

        if frame_detections.shape[0] < valid_markers:
            continue

        world_points = []
        surface_points = []

        if detection_mode == "marker":
            for _, detection in frame_detections.iterrows():
                marker_name = detection["marker name"]
                if marker_name not in layout_df["marker name"].values:
                    continue

                corners_detected = np.array(
                    [
                        [detection["top left x [px]"], detection["top left y [px]"]],
                        [
                            detection["top right x [px]"],
                            detection["top right y [px]"],
                        ],
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
                ref_corners = layout_df.loc[
                    layout_df["marker name"] == marker_name, "marker_corners"
                ].values[0]

                if corners_detected.shape != (4, 2) or ref_corners.shape != (4, 2):
                    raise ValueError(
                        "Marker corners must have shape (4, 2), got "
                        f"{corners_detected.shape} and {ref_corners.shape}"
                    )

                world_points.extend(corners_detected)
                surface_points.extend(ref_corners)
        else:
            for _, detection in frame_detections.iterrows():
                corners_detected = np.asarray(detection["corners"], dtype=np.float32)
                if corners_detected.shape != (4, 2):
                    raise ValueError(
                        f"Detected corners must have shape (4, 2), got {corners_detected.shape}"
                    )

                if layout_by_tag is not None:
                    if "tag_id" not in detection:
                        continue
                    tag_id = int(detection["tag_id"])
                    if tag_id not in layout_by_tag:
                        continue
                    ref_corners = layout_by_tag[tag_id]
                else:
                    ref_corners = base_corners

                world_points.extend(corners_detected)
                surface_points.extend(ref_corners)

        if len(world_points) < required_points:
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


def _prepare_marker_layout(marker_layout: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(marker_layout, pd.DataFrame):
        raise ValueError("marker_layout must be a DataFrame for marker detections.")
    if not MARKERS_LAYOUT_COLUMNS.issubset(marker_layout.columns):
        raise ValueError(
            "marker_layout must contain the following columns: "
            f"{', '.join(MARKERS_LAYOUT_COLUMNS)}"
        )

    marker_layout = marker_layout.copy()
    marker_layout["marker_corners"] = marker_layout.apply(
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


def _prepare_corner_layout(
    surface_layout: pd.DataFrame | np.ndarray,
) -> tuple[Optional[pd.DataFrame], Optional[dict[int, np.ndarray]], Optional[np.ndarray]]:
    if isinstance(surface_layout, np.ndarray):
        base_corners = np.asarray(surface_layout, dtype=np.float32)
        if base_corners.shape != (4, 2):
            raise ValueError("surface_layout array must have shape (4, 2).")
        return None, None, base_corners

    if not isinstance(surface_layout, pd.DataFrame):
        raise ValueError("surface_layout must be a DataFrame or 4x2 numpy array.")

    if "corners" not in surface_layout.columns:
        raise ValueError("surface_layout must contain a 'corners' column.")

    if "tag_id" in surface_layout.columns:
        layout_by_tag = {}
        for _, row in surface_layout.iterrows():
            tag_id = int(row["tag_id"])
            corners = np.asarray(row["corners"], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError("Each layout 'corners' entry must be shape (4, 2).")
            layout_by_tag[tag_id] = corners
        return surface_layout, layout_by_tag, None

    if len(surface_layout) != 1:
        raise ValueError(
            "surface_layout must have one row or include a 'tag_id' column when using corner detections."
        )

    base_corners = np.asarray(surface_layout["corners"].iloc[0], dtype=np.float32)
    if base_corners.shape != (4, 2):
        raise ValueError("surface_layout 'corners' must have shape (4, 2).")
    return surface_layout, None, base_corners

