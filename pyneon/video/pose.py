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
    Estimate the camera pose for each frame by solving a Perspective-n-Point (PnP)
    problem based on marker detections (AprilTag, ArUco, or screen).

    The function automatically adapts to the detection method provided in
    `all_detections["method"]`. For planar screens, if `tag_locations_df` is not
    provided, reference 3D positions are inferred from `screen_size`.

    Parameters
    ----------
    video : SceneVideo
        SceneVideo instance providing the frames' timestamps and intrinsic
        calibration matrices (`camera_matrix`, `dist_coeffs`).
    tag_locations_df : pandas.DataFrame, optional
        Table describing the 3D world coordinates of each marker or screen.
        Expected columns for tag-based detections:
            - "tag_id" or "marker_id" : int
            - "pos_vec"  : list[float] (3D position of tag center)
            - "norm_vec" : list[float] (surface normal)
            - "size"     : float (edge length)
        For screen-based detection, this can be omitted if `screen_size` is given.
    all_detections : pandas.DataFrame, optional
        Per-frame detections returned by any of:
            - `detect_apriltags()`
            - `detect_aruco()`
            - `detect_screen_corners()`
        Must contain columns:
            - "frame_idx" : int
            - "corners"   : ndarray (4, 2)
            - "tag_id"    : int (or 0 for screen)
            - "method"    : str (e.g. "apriltag", "aruco", "screen")
    screen_size : tuple[int, int], optional
        Physical or pixel dimensions of the screen (width, height). Used when
        `method == "screen"` and `tag_locations_df` is not provided.

    Returns
    -------
    pandas.DataFrame
        One row per frame with columns:
            - "frame_idx"
            - "translation_vector" : ndarray (3,)
            - "rotation_vector"    : ndarray (3,)
            - "camera_pos"         : ndarray (3,)
            - "method"             : str
    """

    # ------------------------------------------------------------------
    # 1. Prepare detections
    # ------------------------------------------------------------------
    if all_detections is None or all_detections.empty:
        raise ValueError("`all_detections` must be provided and non-empty.")

    # Extract method from detections
    method = (
        all_detections["method"].iloc[0]
        if "method" in all_detections.columns
        else "apriltag"
    )

    # ------------------------------------------------------------------
    # 2. Handle camera intrinsics
    # ------------------------------------------------------------------
    camera_matrix = getattr(video, "camera_matrix", None)
    dist_coeffs = getattr(video, "dist_coeffs", None)
    if camera_matrix is None:
        raise ValueError("`video` must provide a valid `camera_matrix`.")
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # 3. Build tag info lookup
    # ------------------------------------------------------------------
    tag_info = {}

    if method in ("apriltag", "aruco"):
        if tag_locations_df is None or tag_locations_df.empty:
            raise ValueError(
                f"`tag_locations_df` is required for method='{method}'. "
                "Provide tag 3D locations and orientations."
            )

        # Normalize column names
        if (
            "marker_id" in tag_locations_df.columns
            and "tag_id" not in tag_locations_df.columns
        ):
            tag_locations_df = tag_locations_df.rename(columns={"marker_id": "tag_id"})

        required = {"tag_id", "pos_vec", "norm_vec", "size"}
        missing = required - set(tag_locations_df.columns)
        if missing:
            raise ValueError(f"tag_locations_df is missing: {missing}")

        for _, row in tag_locations_df.iterrows():
            tag_id = int(row["tag_id"])
            center = np.asarray(row["pos_vec"], dtype=np.float32)
            normal = np.asarray(row["norm_vec"], dtype=np.float32)
            normal /= np.linalg.norm(normal)
            half = float(row["size"]) / 2.0
            tag_info[tag_id] = (center, normal, half)

    elif method == "screen":
        # For screens, assume a single flat rectangle in XY-plane
        if tag_locations_df is not None and not tag_locations_df.empty:
            # Use provided corners or plane definition
            tag_info[0] = tag_locations_df
        elif screen_size is not None:
            w, h = screen_size
            # Define a simple square in z=0 plane centered at origin
            # Using arbitrary scale (1 m per 1000 px if unknown)
            scale = 0.001
            half_w, half_h = (w * scale) / 2, (h * scale) / 2
            corners_3d = np.array(
                [
                    [-half_w, -half_h, 0],
                    [half_w, -half_h, 0],
                    [half_w, half_h, 0],
                    [-half_w, half_h, 0],
                ],
                dtype=np.float32,
            )
            tag_info[0] = corners_3d
        else:
            raise ValueError(
                "For method='screen', provide either `tag_locations_df` or `screen_size`."
            )
    else:
        raise ValueError(f"Unsupported detection method: '{method}'")

    # ------------------------------------------------------------------
    # 4. Iterate over frames and solve PnP
    # ------------------------------------------------------------------
    results = []
    for frame_idx in all_detections["frame_idx"].unique():
        det_frame = all_detections.loc[all_detections["frame_idx"] == frame_idx]
        if det_frame.empty:
            continue

        object_pts, image_pts = [], []

        for _, det in det_frame.iterrows():
            tag_id = int(det.get("tag_id", 0))
            corners_2d = np.asarray(det["corners"], dtype=np.float32)

            # For screen: use single planar surface definition
            if method == "screen":
                if 0 not in tag_info:
                    continue
                corners_3d = np.asarray(tag_info[0], dtype=np.float32)

            else:  # ArUco / AprilTag
                if tag_id not in tag_info:
                    continue
                center3d, normal, half = tag_info[tag_id]

                # Construct local tag coordinate basis
                Z = normal
                ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if np.allclose(Z, ref) or np.allclose(Z, -ref):
                    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                X = np.cross(Z, ref)
                if np.linalg.norm(X) < 1e-9:
                    ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    X = np.cross(Z, ref)
                X /= np.linalg.norm(X)
                Y = np.cross(Z, X)

                # 3D tag corners in world coordinates
                corners_3d = np.vstack(
                    [
                        center3d + (-half) * X + (-half) * Y,
                        center3d + (half) * X + (-half) * Y,
                        center3d + (half) * X + (half) * Y,
                        center3d + (-half) * X + (half) * Y,
                    ]
                ).astype(np.float32)

            if corners_3d.shape != (4, 3):
                continue

            object_pts.append(corners_3d)
            image_pts.append(corners_2d)

        if not object_pts:
            continue

        object_pts = np.vstack(object_pts)
        image_pts = np.vstack(image_pts)

        # ------------------------------------------------------------------
        # 5. Solve PnP
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
                "translation_vector": t_vec.reshape(-1),
                "rotation_vector": r_vec.reshape(-1),
                "camera_pos": cam_pos.reshape(-1),
                "method": method,
            }
        )

    # ------------------------------------------------------------------
    # 6. Return DataFrame
    # ------------------------------------------------------------------
    return pd.DataFrame(
        results,
        columns=[
            "frame_idx",
            "translation_vector",
            "rotation_vector",
            "camera_pos",
            "method",
        ],
    )
