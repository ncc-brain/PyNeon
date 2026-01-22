from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .video import Video


def estimate_camera_pose(
    video: "Video",
    tag_locations_df: pd.DataFrame,
    all_detections: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estimate the camera pose for every frame by solving a Perspective-n-Point
    (PnP) problem based on marker detections.

    Parameters
    ----------
    video :
        ``SceneVideo`` instance providing the frames' timestamps and the
        intrinsic matrices ``camera_matrix`` and ``dist_coeffs``.
    tag_locations_df :
        pandas.DataFrame describing the world-coordinates of each marker.
        Required columns::

            "tag_id"   : int
            "pos_vec"  : list[float] length 3, [x, y, z] position of the tag
            "norm_vec" : list[float] length 3, [nx, ny, nz] front-face normal
            "size"     : float, edge length in meters
    all_detections : Stream or pandas.DataFrame, optional
        Per-frame tag detections. If *None* the function calls ``detect_markers(video)``.
        Expected columns::

            "frame id" : int
            "tag id"   : int
            "corner 0 x [px]", "corner 0 y [px]": First corner coordinates
            "corner 1 x [px]", "corner 1 y [px]": Second corner coordinates
            "corner 2 x [px]", "corner 2 y [px]": Third corner coordinates
            "corner 3 x [px]", "corner 3 y [px]": Fourth corner coordinates

    Returns
    -------
    pandas.DataFrame
        One row per processed frame with columns::

            "frame id"          : int
            "translation_vector" : ndarray (3,)
            "rotation_vector"    : ndarray (3,)
            "camera_pos"         : ndarray (3,) camera position in world coord.
    """

    # ------------------------------------------------------------------ prepare detections
    if all_detections is None:
        from .marker_mapping import detect_markers  # local import to avoid cycle

        det_stream = detect_markers(video)
        detections_df = det_stream.data
    else:
        detections_df = getattr(all_detections, "data", all_detections)

    if detections_df.empty:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "translation_vector",
                "rotation_vector",
                "camera_pos",
            ]
        )

    # ------------------------------------------------------------------ camera intrinsics
    camera_matrix = video.camera_matrix
    dist_coeffs = video.dist_coeffs
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # ------------------------------------------------------------------ build lookup: tag_id -> (center, normal, half_size)
    tag_info = {}
    required = {"tag_id", "pos_vec", "norm_vec", "size"}
    if not required.issubset(tag_locations_df.columns):
        missing = required - set(tag_locations_df.columns)
        raise ValueError(f"tag_locations_df is missing: {missing}")

    for _, row in tag_locations_df.iterrows():
        tag_id = int(row["tag_id"])
        center = np.asarray(row["pos_vec"], dtype=np.float32)
        normal = np.asarray(row["norm_vec"], dtype=np.float32)
        normal /= np.linalg.norm(normal)  # normalize
        half = float(row["size"]) / 2.0
        tag_info[tag_id] = (center, normal, half)

    # ------------------------------------------------------------------ iterate over frames
    results = []
    for frame in detections_df["frame id"].unique():
        det_frame = detections_df.loc[detections_df["frame id"] == frame]
        if det_frame.empty:
            continue

        object_pts, image_pts = [], []

        for _, det in det_frame.iterrows():
            # Reconstruct corners array from individual columns
            corners_2d = np.array(
                [
                    [det["corner 0 x [px]"], det["corner 0 y [px]"]],
                    [det["corner 1 x [px]"], det["corner 1 y [px]"]],
                    [det["corner 2 x [px]"], det["corner 2 y [px]"]],
                    [det["corner 3 x [px]"], det["corner 3 y [px]"]],
                ],
                dtype=np.float32,
            )

            tid = int(det["tag id"])
            if tid not in tag_info:
                continue

            center3d, normal, half = tag_info[tid]
            # build orthonormal basis (X, Y, Z)
            Z = normal
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if np.allclose(Z, ref) or np.allclose(Z, -ref):
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            X = np.cross(Z, ref)
            if np.linalg.norm(X) < 1e-9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                X = np.cross(Z, ref)
            X = X / np.linalg.norm(X)
            Y = np.cross(Z, X)

            # 3-D corners: BL, BR, TR, TL
            obj_corners = np.vstack(
                [
                    center3d + (-half) * X + (-half) * Y,
                    center3d + (half) * X + (-half) * Y,
                    center3d + (half) * X + (half) * Y,
                    center3d + (-half) * X + (half) * Y,
                ]
            ).astype(np.float32)

            object_pts.append(obj_corners)
            image_pts.append(corners_2d)

        if not object_pts:
            continue

        object_pts = np.vstack(object_pts)
        image_pts = np.vstack(image_pts)

        ok, r_vec, t_vec = cv2.solvePnP(
            object_pts,
            image_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        R, _ = cv2.Rodrigues(r_vec)
        cam_pos = -R.T @ t_vec

        results.append(
            {
                "frame id": int(frame),
                "translation_vector": t_vec.reshape(-1),
                "rotation_vector": r_vec.reshape(-1),
                "camera_pos": cam_pos.reshape(-1),
            }
        )

    return pd.DataFrame(
        results,
        columns=["frame id", "translation_vector", "rotation_vector", "camera_pos"],
    )
