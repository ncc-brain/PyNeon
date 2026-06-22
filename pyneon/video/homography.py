import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.docstring_templating import fill_doc
from .utils import _validate_contour_layout, _validate_marker_layout


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

        {marker_layout_table}

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
    {homographies}

    Examples
    --------
    Compute homographies from marker detections:

    >>> detections = video.detect_markers("36h11")
    >>> layout = pd.DataFrame({{
    ...     "marker name": ["36h11_0", "36h11_1"],
    ...     "size": [100, 100],
    ...     "center x": [200, 400],
    ...     "center y": [200, 200],
    ... }})
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
        # Compute corner coordinates for each marker in the layout using vectorized arrays.
        layout = layout.copy()
        center_x = layout["center x"].to_numpy()
        center_y = layout["center y"].to_numpy()
        half_size = layout["size"].to_numpy() / 2
        corners_array = np.stack(
            [
                np.column_stack((center_x - half_size, center_y - half_size)),
                np.column_stack((center_x + half_size, center_y - half_size)),
                np.column_stack((center_x + half_size, center_y + half_size)),
                np.column_stack((center_x - half_size, center_y + half_size)),
            ],
            axis=1,
        )
        layout["corners"] = list(corners_array)
        # Construct a lookup dictionary with marker name being key and corners being value
        surface_pts_lookup = {
            marker_name: corners
            for marker_name, corners in layout[["marker name", "corners"]].itertuples(
                index=False, name=None
            )
        }
    else:
        _validate_contour_layout(layout)
        surface_pts_lookup = {"contour_0": layout}

    corner_columns = [
        "top left x [px]",
        "top left y [px]",
        "top right x [px]",
        "top right y [px]",
        "bottom right x [px]",
        "bottom right y [px]",
        "bottom left x [px]",
        "bottom left y [px]",
    ]

    homography_per_frame = {}
    grouped_detections = detection_df.groupby(level=0, sort=False)

    for ts, frame_detections in tqdm(
        grouped_detections,
        total=grouped_detections.ngroups,
        desc="Computing surface-mapping homographies",
    ):
        if isinstance(frame_detections.index, pd.MultiIndex):
            frame_detections = frame_detections.droplevel(0)

        if is_marker_detection and len(frame_detections) < min_markers:
            continue

        camera_pts_all = []
        surface_pts_all = []

        corner_rows = frame_detections[corner_columns].itertuples(
            index=False, name=None
        )
        if is_marker_detection:
            detection_iter = zip(
                frame_detections["marker name"].to_numpy(),
                corner_rows,
            )
        else:
            detection_iter = (("contour_0", corners) for corners in corner_rows)

        for name, corners in detection_iter:
            camera_pts = np.array(
                [
                    [corners[0], corners[1]],
                    [corners[2], corners[3]],
                    [corners[4], corners[5]],
                    [corners[6], corners[7]],
                ],
                dtype=np.float32,
            )
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
