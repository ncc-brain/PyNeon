import cv2
import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .video import SceneVideo


def estimate_camera_pose(
    video: "SceneVideo",
    tag_locations_df: Optional[pd.DataFrame] = None,
    all_detections: Optional[pd.DataFrame] = None,
    screen_size: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Estimate the camera pose (translation, rotation, and position) for each frame
    using 2D–3D correspondences from AprilTag, ArUco, or screen detections.

    The function automatically adapts to the detection method in
    `all_detections["method"]`.  For planar screen detections, 3D reference points
    are generated from `screen_size` if no tag location table is provided.

    Parameters
    ----------
    video : SceneVideo
        Video object providing camera intrinsics (`camera_matrix`, `dist_coeffs`)
        and timestamps (`video.ts`).
    tag_locations_df : pandas.DataFrame, optional
        World-space information for tag-based detections.
        Required columns:
            - "tag_id" or "marker_id" : int
            - "pos_vec"  : list[float] (3D center position)
            - "norm_vec" : list[float] (surface normal)
            - "size"     : float (edge length)
        Ignored for method='screen' if `screen_size` is supplied.
    all_detections : pandas.DataFrame, optional
        Per-frame detections from `detect_apriltags`, `detect_aruco`,
        or `detect_screen_corners`.  Must contain:
            - "frame_idx" : int
            - "corners"   : ndarray (4,2)
            - "tag_id"    : int (0 for screen)
            - "method"    : str
    screen_size : tuple[int, int], optional
        Screen dimensions (width, height) in pixels or physical units.
        Used only for `method='screen'` when no tag_locations_df is given.

    Returns
    -------
    pandas.DataFrame
        One row per processed frame with:
            - 'frame_idx'          : int
            - 'translation_vector' : np.ndarray (3,)
            - 'rotation_vector'    : np.ndarray (3,)
            - 'camera_pos'         : np.ndarray (3,)
            - 'method'             : str

    Notes
    -----
    - Uses OpenCV's `solvePnP` to compute camera extrinsics from all visible
        markers per frame.
    - For tag-based detections, 3D corners are built from each tag’s position,
        normal vector, and edge length.
    - For screen detections, a planar rectangle centered at the origin is used
        unless explicit 3D coordinates are supplied.
    """

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------
    if all_detections is None or all_detections.empty:
        raise ValueError("`all_detections` must be provided and non-empty.")

    method = all_detections.get("method", pd.Series(["apriltag"])).iloc[0]

    # ------------------------------------------------------------------
    # 2. Camera intrinsics
    # ------------------------------------------------------------------
    camera_matrix = getattr(video, "camera_matrix", None)
    if camera_matrix is None:
        raise ValueError("`video` must provide a valid `camera_matrix`.")
    dist_coeffs = getattr(video, "dist_coeffs", None)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), np.float32)

    # ------------------------------------------------------------------
    # 3. Build 3D reference geometry
    # ------------------------------------------------------------------
    tag_info = {}

    if method in ("apriltag", "aruco"):
        # Normalize column names
        if tag_locations_df is None or tag_locations_df.empty:
            raise ValueError(f"`tag_locations_df` required for method='{method}'.")
        if (
            "marker_id" in tag_locations_df.columns
            and "tag_id" not in tag_locations_df.columns
        ):
            tag_locations_df = tag_locations_df.rename(columns={"marker_id": "tag_id"})

        required = {"tag_id", "pos_vec", "norm_vec", "size"}
        missing = required - set(tag_locations_df.columns)
        if missing:
            raise ValueError(f"tag_locations_df missing columns: {missing}")

        for _, row in tag_locations_df.iterrows():
            tag_id = int(row["tag_id"])
            center = np.asarray(row["pos_vec"], np.float32)
            normal = np.asarray(row["norm_vec"], np.float32)
            normal /= np.linalg.norm(normal)
            half = float(row["size"]) / 2.0
            tag_info[tag_id] = (center, normal, half)

    elif method == "screen":
        if tag_locations_df is not None and not tag_locations_df.empty:
            tag_info[0] = tag_locations_df
        elif screen_size is not None:
            w, h = screen_size
            scale = 0.001  # 1 m per 1000 px if physical scale unknown
            half_w, half_h = (w * scale) / 2, (h * scale) / 2
            tag_info[0] = np.array(
                [
                    [-half_w, -half_h, 0],
                    [half_w, -half_h, 0],
                    [half_w, half_h, 0],
                    [-half_w, half_h, 0],
                ],
                np.float32,
            )
        else:
            raise ValueError(
                "For 'screen', provide `tag_locations_df` or `screen_size`."
            )
    else:
        raise ValueError(f"Unsupported detection method: '{method}'")

    # ------------------------------------------------------------------
    # 4. Iterate over frames
    # ------------------------------------------------------------------
    results = []
    for frame_idx, frame_dets in all_detections.groupby("frame_idx"):
        object_pts, image_pts = [], []

        for _, det in frame_dets.iterrows():
            tag_id = int(det.get("tag_id", 0))
            corners_2d = np.asarray(det["corners"], np.float32)

            if method == "screen":
                corners_3d = np.asarray(tag_info[0], np.float32)
            else:
                if tag_id not in tag_info:
                    continue
                center3d, normal, half = tag_info[tag_id]

                # Local tag basis
                Z = normal
                ref = np.array([0, 0, 1], np.float32)
                if np.allclose(Z, ref) or np.allclose(Z, -ref):
                    ref = np.array([1, 0, 0], np.float32)
                X = np.cross(Z, ref)
                if np.linalg.norm(X) < 1e-9:
                    ref = np.array([0, 1, 0], np.float32)
                    X = np.cross(Z, ref)
                X /= np.linalg.norm(X)
                Y = np.cross(Z, X)

                corners_3d = np.vstack(
                    [
                        center3d + (-half) * X + (-half) * Y,
                        center3d + (half) * X + (-half) * Y,
                        center3d + (half) * X + (half) * Y,
                        center3d + (-half) * X + (half) * Y,
                    ]
                ).astype(np.float32)

            object_pts.append(corners_3d)
            image_pts.append(corners_2d)

        if not object_pts:
            continue

        object_pts = np.vstack(object_pts)
        image_pts = np.vstack(image_pts)

        # ------------------------------------------------------------------
        # 5. Solve PnP for this frame
        # ------------------------------------------------------------------
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
                "frame_idx": int(frame_idx),
                "translation_vector": t_vec.flatten(),
                "rotation_vector": r_vec.flatten(),
                "camera_pos": cam_pos.flatten(),
                "method": method,
            }
        )

    return pd.DataFrame(results)
