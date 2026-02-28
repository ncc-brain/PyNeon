import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .utils import _validate_contour_layout, _validate_marker_layout


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
    Compute a per-frame homography (3x3 matrix) from detections to
    a surface coordinate system.

    Parameters
    ----------
    detections : Stream
        Stream containing per-detection marker/contour coordinates returned
        by :meth:`Video.detect_markers` or :meth:`Video.detect_contour`.
    layout : pd.DataFrame or np.ndarray
        Layout of markers/contour to provide reference surface coordinates for homography computation.
        The expected format depends on the type of detections:

        **Marker detections**: provide a DataFrame (can be visually checked with
        :func:`pyneon.plot_marker_layout`) with following columns:

            %(marker_layout_table)s

        **Contour detections**: provide a 2D numpy array of shape (4, 2)
        containing the surface coordinates of the contour corners in the following order:
        top-left, top-right, bottom-right, bottom-left.

    min_markers : int, optional
        Minimum number of marker detections required in a frame to compute a
        homography when using marker detections. Frames with fewer detections are
        skipped. Defaults to 2.
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
        Confidence level, between 0 and 1. Defaults to 0.995.

    Returns
    -------
    %(homographies)s

    Examples
    --------
    Compute homographies from marker detections:

    >>> detections = video.detect_markers("36h11")
    >>> layout = pd.DataFrame({
    ...     "marker name": ["36h11_0", "36h11_1"],
    ...     "size": [100, 100],
    ...     "center x": [200, 400],
    ...     "center y": [200, 200],
    ... })
    >>> homographies = find_homographies(detections, layout)

    Compute homographies from contour detections:

    >>> detections = video.detect_contour()
    >>> layout = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> homographies = find_homographies(detections, layout)
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
        surface_pts_lookup = {
            row["marker name"]: row["corners"] for _, row in layout.iterrows()
        }
    else:
        _validate_contour_layout(layout)
        surface_pts_lookup = {"contour_0": layout}

    homography_per_frame = {}
    unique_timestamps = detection_df.index.unique()

    for ts in tqdm(unique_timestamps, desc="Computing surface-mapping homographies"):
        frame_detections = detection_df.loc[ts]
        if isinstance(frame_detections, pd.Series):
            frame_detections = frame_detections.to_frame().T

        if is_marker_detection and len(frame_detections) < min_markers:
            continue

        camera_pts_all = []
        surface_pts_all = []

        for _, detection in frame_detections.iterrows():
            camera_pts = _reshape_corners(detection)
            name = detection["marker name"] if is_marker_detection else "contour_0"
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
