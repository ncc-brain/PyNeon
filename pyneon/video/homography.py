import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional
from pyneon.stream import Stream

if TYPE_CHECKING:
    from .video import SceneVideo

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_homographies(
    video: "SceneVideo",
    detection_df: pd.DataFrame,
    tag_info: Optional[pd.DataFrame] = None,
    frame_size: tuple[int, int] = (1920, 1080),
    coordinate_system: str = "opencv",
    skip_frames: int = 1,
    undistort: bool = True,
    sample_ts: Optional[np.ndarray] = None,
    settings: Optional[dict] = None,
    return_diagnostics: bool = True,
) -> pd.DataFrame:
    """
    Compute a homography for each frame using available AprilTag detections.

    This function identifies all markers detected in a given frame, looks up their
    "ideal" (reference) positions from `tag_info`, and calls OpenCV's
    `cv2.findHomography` to compute a 3x3 transformation matrix mapping from
    detected corners in the video image to the reference plane (e.g., surface coordinates).

    If the coordinate system is "psychopy", corners in both `tag_info` and
    `detection_df` are first converted to an OpenCV-like pixel coordinate system.
    If `undistort=True` and camera intrinsics are available in the `video` object,
    the marker corners are also undistorted.

    The optional `homography_settings` dictionary allows customizing parameters like
    RANSAC thresholds and maximum iterations. The default is an OpenCV RANSAC method
    with moderate thresholds.

    Parameters
    ----------
    video : SceneVideo
        An object containing camera intrinsics (camera_matrix, dist_coeffs) and possibly timestamps.
        If `undistort=True`, these intrinsics are used to undistort marker corners.
    detection_df : pandas.DataFrame
        Must contain:
        - 'frame_idx': int
        - 'tag_id': int
        - 'corners': np.ndarray of shape (4, 2) in video or PsychoPy coordinates
    tag_info : pandas.DataFrame
        Must contain:
        - 'marker_id' (or 'tag_id'): int
        - 'marker_corners': np.ndarray of shape (4, 2) giving the reference positions
            for each corner (e.g., on a surface plane)
    frame_size : (width, height)
        The pixel resolution of the video frames. Used if `coordinate_system="psychopy"`
        to convert from PsychoPy to OpenCV-style coordinates.
    coordinate_system : str, optional
        One of {"opencv", "psychopy"}. If "psychopy", corners in `detection_df` and
        `tag_info` are converted to OpenCV pixel coords before the homography is computed.
        Default is "opencv".
    skip_frames : int, optional
        If > 1, the function will compute homographies only for every Nth frame.
        E.g., skip_frames=5 will compute homographies for frames 0, 5, 10, 15, etc.
        Must match with the `skip_frames` used in `detect_apriltags`.
    undistort : bool, optional
        Whether to undistort marker corners using the camera intrinsics in `video`.
        Default is True.
    settings : dict, optional
        A dictionary of parameters passed to `cv2.findHomography`. For example:
        {
            "method": cv2.RANSAC,
            "ransacReprojThreshold": 2.0,
            "maxIters": 500,
            "confidence": 0.98,
        }
        The defaults are set to a moderate RANSAC approach.
    return_diagnostics : bool, optional
        If True, return additional diagnostic information such as the inlier mask from RANSAC.
        Default is True.

    Returns
    -------
    dict
        A dictionary mapping each frame index (`frame_idx`: int) to its corresponding
        homography matrix (3x3 NumPy array) or None if insufficient markers or points
        were available to compute a valid homography.
    """

    detection_df = detection_df.copy()

    if "marker_id" not in tag_info.columns and "tag_id" in tag_info.columns:
        tag_info = tag_info.rename(columns={"tag_id": "marker_id"})

    method = (
        detection_df["method"].iloc[0]
        if "method" in detection_df.columns
        else "apriltag"
    )

    # -----------------------------------------------------------------
    # 0. Handle special case: screen-based method with no tag_info
    # -----------------------------------------------------------------

    if method == "screen" and tag_info.empty:
        if frame_size is None:
            raise ValueError(
                "For method='screen', frame_size=(width,height) must be set."
            )
        w, h = frame_size
        tag_info = pd.DataFrame(
            [
                {
                    "marker_id": 0,
                    "marker_corners": np.array(
                        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
                    ),
                }
            ]
        )
    elif method == "apriltag" or method == "aruco":
        if tag_info is None or tag_info.empty:
            raise ValueError(
                "tag_info DataFrame must be provided for AprilTag or ArUco methods."
            )
    # -----------------------------------------------------------------
    # 1. Convert from PsychoPy coords to OpenCV if necessary
    # -----------------------------------------------------------------
    if coordinate_system.lower() == "psychopy":
        # Example transform function
        def psychopy_coords_to_opencv(coords, frame_size):
            w, h = frame_size
            coords = np.array(coords)  # Convert list to ndarray
            x_opencv = coords[:, 0] + (w / 2)
            y_opencv = (h / 2) - coords[:, 1]
            return np.column_stack(
                (x_opencv, y_opencv)
            ).tolist()  # Convert back to list

        # Convert the reference corners in tag_info
        def convert_marker_corners(c):
            return psychopy_coords_to_opencv(c, frame_size)

        tag_info["marker_corners"] = tag_info["marker_corners"].apply(
            convert_marker_corners
        )

    # -----------------------------------------------------------------
    # 2. Undistort corners & gaze if desired
    # -----------------------------------------------------------------
    camera_matrix = getattr(video, "camera_matrix", None)
    dist_coeffs = getattr(video, "dist_coeffs", None)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    if undistort and camera_matrix is not None:

        def undistort_points(
            points: np.ndarray, K: np.ndarray, D: np.ndarray
        ) -> np.ndarray:
            pts_for_cv = points.reshape((-1, 1, 2)).astype(np.float32)
            undist = cv2.undistortPoints(pts_for_cv, K, D)
            # undistortPoints outputs normalized coords => multiply back by K
            ones = np.ones((undist.shape[0], 1, 1), dtype=np.float32)
            undist_hom = np.concatenate([undist, ones], axis=2)
            pixel = np.einsum("ij,nkj->nki", K, undist_hom)
            pixel = pixel[:, 0, :2] / pixel[:, 0, 2:]
            return pixel

        # Undistort detection corners
        def undistort_detection_corners(row):
            c = np.array(row["corners"])
            return undistort_points(c, camera_matrix, dist_coeffs)

        detection_df["corners"] = detection_df.apply(
            undistort_detection_corners, axis=1
        )

    # -----------------------------------------------------------------
    # 3. Compute a homography for each frame using *all* tag detections
    # -----------------------------------------------------------------
    default_settings = {
        "method": cv2.LMEDS  # Disable RANSAC completely
    }

    # Merge user-provided settings with defaults
    if settings is not None:
        default_settings.update(settings)

    frames = detection_df["frame_idx"].unique()
    homography_for_frame = {}

    marker_dict = {}
    for _, row in tag_info.iterrows():
        marker_dict[row["marker_id"]] = np.array(
            row["marker_corners"], dtype=np.float32
        )

    frames = detection_df["frame_idx"].unique()
    homography_for_frame = {}

    for frame in tqdm(frames, desc="Computing homographies for frames"):
        frame_detections = detection_df.loc[detection_df["frame_idx"] == frame]
        if frame_detections.empty:
            homography_for_frame[frame] = None
            continue

        world_points = []  # from the camera's perspective (detected corners)
        surface_points = []  # from the reference plane or "ideal" positions

        for _, detection in frame_detections.iterrows():
            tag_id = detection["tag_id"]

            if tag_id not in marker_dict:
                # no reference corners for this tag
                continue

            corners_detected = np.array(detection["corners"], dtype=np.float32)
            ref_corners = marker_dict[tag_id]

            # optional shape check:
            if corners_detected.shape != (4, 2) or ref_corners.shape != (4, 2):
                continue

            # Extend our list of corner correspondences
            world_points.extend(corners_detected)  # add 4 corner coords
            surface_points.extend(ref_corners)  # add 4 reference coords

        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
        surface_points = np.array(surface_points, dtype=np.float32).reshape(-1, 2)

        if len(world_points) < 4:
            # Not enough corners to compute a homography
            homography_for_frame[frame] = None
            continue

        H, mask = cv2.findHomography(world_points, surface_points, **default_settings)
        homography_for_frame[frame] = H

    if skip_frames != 1:
        # Upsample the homographies to fill in skipped frames
        max_frame = max(frames)
        homography_for_frame = _upsample_homographies(homography_for_frame, max_frame)

    # Get timestamps for each frame_idx
    frame_idx_to_ts = dict(zip(range(len(video.ts)), video.ts))

    if not return_diagnostics:
        records = [
            {
                "timestamp [ns]": frame_idx_to_ts[frame],
                "frame_idx": frame,
                "homography": H,
            }
            for frame, H in homography_for_frame.items()
            if frame in frame_idx_to_ts
        ]
    else:
        records = [
            {
                "timestamp [ns]": frame_idx_to_ts[frame],
                "frame_idx": frame,
                "homography": H,
                "mask": mask,
            }
            for frame, H in homography_for_frame.items()
            if frame in frame_idx_to_ts
        ]

    df = pd.DataFrame.from_records(records)
    df = df.set_index("timestamp [ns]")

    if sample_ts is not None:
        df = _sample_homography_to_ts(df, sample_ts)

    return df


def gaze_on_surface(
    gaze_df: pd.DataFrame,
    homographies: pd.DataFrame,
    sample_ts: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Apply per-frame homographies to gaze points to transform them into a new coordinate system.

    Parameters
    ----------
    gaze_df : pandas.DataFrame
        DataFrame containing gaze points with columns:
        - 'frame_idx': int, the frame index
        - 'x', 'y': float, the gaze coordinates in the original coordinate system.
    homography_for_frame : dict
        A dictionary mapping frame indices to 3x3 homography matrices.

    Returns
    -------
    pandas.DataFrame
        A copy of `gaze_df` with additional columns:
        - 'x_trans', 'y_trans': the transformed gaze coordinates.
    """
    gaze_df = gaze_df.copy()
    gaze_df["x_trans"] = np.nan
    gaze_df["y_trans"] = np.nan

    if sample_ts is not None:
        # 1) interpolate gaze to sample_ts
        gaze_stream = Stream(gaze_df)
        gaze_interp = gaze_stream.interpolate(new_ts=sample_ts)
        gaze_df = gaze_interp.data

        # 2) must be 1:1 with homographies by index (timestamps)
        if not np.array_equal(gaze_df.index.values, homographies.index.values):
            raise ValueError(
                "Interpolated gaze timestamps do not match homography timestamps."
            )

        # 3) join homographies by index (timestamp)
        merged = gaze_df.join(homographies[["homography"]], how="left")

    else:
        # join by frame_idx (left keep order)
        merged = gaze_df.merge(
            homographies[["frame_idx", "homography"]],
            on="frame_idx",
            how="left",
            sort=False,
        )

    # --- Vectorized per-row transform ---
    # rows with a valid (3x3) homography
    H_col = merged["homography"]
    mask = H_col.notna()

    # homogeneous input points (N, 3)
    pts = merged[["gaze x [px]", "gaze y [px]"]].to_numpy(dtype=float)
    N = len(pts)
    pts_h = np.empty((N, 3), dtype=float)
    pts_h[:, :2] = pts
    pts_h[:, 2] = 1.0

    # stack per-row homographies only for valid rows
    if mask.any():
        H_stack = np.stack(H_col[mask].to_numpy())  # (M, 3, 3)
        pts_sel = pts_h[mask]  # (M, 3)

        # batch multiply: (M,3,3) · (M,3) -> (M,3)
        out = np.einsum("nij,nj->ni", H_stack, pts_sel)  # (M,3)

        # inhomogeneous divide
        w = out[:, 2]
        xw = out[:, 0] / w
        yw = out[:, 1] / w

        x_trans = np.full(N, np.nan, dtype=float)
        y_trans = np.full(N, np.nan, dtype=float)
        x_trans[mask.values] = xw
        y_trans[mask.values] = yw
    else:
        x_trans = np.full(N, np.nan, dtype=float)
        y_trans = np.full(N, np.nan, dtype=float)

    # write back preserving original gaze_df order
    gaze_df["x_trans"] = x_trans[: len(gaze_df)]
    gaze_df["y_trans"] = y_trans[: len(gaze_df)]
    return gaze_df


def _upsample_homographies(
    homographies_dict: dict[int | np.int64, np.ndarray],
    max_frame: int | np.int64,
) -> dict[int, np.ndarray]:
    """
    Upsample/interpolate homographies for all frames from 0..max_frame, inclusive.
    Assumes homographies_dict contains partial frames (e.g., 0, 10, 20...),
    possibly with np.int64 keys.
    Interpolates linearly between each known pair of frames for each 3x3 entry,
    and then ensures keys are plain Python int.

    Parameters
    ----------
    homographies_dict : dict
        Keys are frame indices (int or np.int64), values are 3x3 np.ndarray (float).
    max_frame : int
        The highest frame index you want to fill in.

    Returns
    -------
    dict[int, numpy.ndarray]
        A dictionary with a 3x3 homography for every frame from 0..max_frame.
        Keys will be standard Python int.
        If a frame is not within the known range, it will be extrapolated
        from the closest known pair.
    """
    # ------------------------------------------------------------------
    # 1) Convert any np.int64 keys to Python int so sorting won't break
    # ------------------------------------------------------------------
    homographies_fixed_keys = {}
    for frame_idx, H in homographies_dict.items():
        homographies_fixed_keys[int(frame_idx)] = H

    # ------------------------------------------------------------------
    # 2) Sort frames that have known homographies
    # ------------------------------------------------------------------
    known_frames = sorted(homographies_fixed_keys.keys())
    if not known_frames:
        # No known frames => return empty dict
        return {}

    upsampled = {}

    # ------------------------------------------------------------------
    # 3) Interpolate between consecutive known frames
    # ------------------------------------------------------------------
    for i in tqdm(range(len(known_frames) - 1), desc="Interpolating homographies"):
        f1 = known_frames[i]
        f2 = known_frames[i + 1]

        H1 = homographies_fixed_keys[f1]  # 3x3
        H2 = homographies_fixed_keys[f2]  # 3x3

        frame_diff = f2 - f1

        # Fill from f1..(f2-1)
        for offset in range(frame_diff):
            current_frame = f1 + offset
            if current_frame > max_frame:
                break

            alpha = offset / float(frame_diff)
            H_interp = (1 - alpha) * H1 + alpha * H2
            upsampled[current_frame] = H_interp

    # Make sure we include the last known frame
    last_frame = known_frames[-1]
    if last_frame <= max_frame:
        upsampled[last_frame] = homographies_fixed_keys[last_frame]

    # ------------------------------------------------------------------
    # 4) Fill frames beyond the last known up to max_frame (extrapolate)
    # ------------------------------------------------------------------
    for f in range(last_frame + 1, max_frame + 1):
        upsampled[f] = homographies_fixed_keys[last_frame]

    # Now, "upsampled" is already using Python int keys,
    # so no further conversion is strictly needed.

    return upsampled


def _sample_homography_to_ts(
    homographies_df: pd.DataFrame,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Upsample/interpolate homographies for arbitrary timestamps.
    Interpolates linearly between each known pair of timestamps for each 3x3 entry.

    Parameters
    ----------
    homographies_df : pd.DataFrame
        DataFrame with index 'timestamp [ns]' and columns 'frame_idx', 'homography'
    timestamps : np.ndarray
        Target timestamps where you want homographies interpolated.

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated homographies for target timestamps.
        Index: 'timestamp [ns]', columns: 'homography'
    """

    if homographies_df.empty:
        return pd.DataFrame(columns=["homography", "frame_idx"]).set_index(
            pd.Index([], name="timestamp [ns]")
        )

    # Ensure sorted by timestamp
    df = homographies_df.sort_index()
    ts_known = df.index.values
    frames_known = df["frame_idx"].to_numpy()

    # Stack to (K, 3, 3); ensure float dtype
    H_known = np.stack(df["homography"].to_numpy()).astype(float)  # (K,3,3)
    K = len(ts_known)

    # Handle degenerate K
    if K == 1:
        H_interp = np.repeat(H_known, len(timestamps), axis=0)
        frames = np.full(len(timestamps), frames_known[0])
        return pd.DataFrame(
            {"homography": list(H_interp), "frame_idx": frames},
            index=pd.Index(timestamps, name="timestamp [ns]"),
        )

    # For each target, find right insertion index
    idx_r = np.searchsorted(ts_known, timestamps, side="right")
    # clamp to [1, K-1] so we always have a left and right
    idx_r_clamped = np.clip(idx_r, 1, K - 1)
    idx_l = idx_r_clamped - 1

    t1 = ts_known[idx_l]
    t2 = ts_known[idx_r_clamped]

    # alpha in [0,1] for interior; set to 0 (left) before first, 1 (right) after last
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (timestamps - t1) / (t2 - t1)
    alpha = np.where(timestamps <= ts_known[0], 0.0, alpha)
    alpha = np.where(timestamps >= ts_known[-1], 1.0, alpha)

    # Gather left/right homographies
    H1 = H_known[idx_l]  # (N,3,3)
    H2 = H_known[idx_r_clamped]  # (N,3,3)

    # Broadcast interpolate
    a = alpha[:, None, None]
    H_interp = (1.0 - a) * H1 + a * H2  # (N,3,3)

    # Choose frame index (nearest by alpha; ties -> right)
    frame_l = frames_known[idx_l]
    frame_r = frames_known[idx_r_clamped]
    frames = np.where(
        timestamps <= ts_known[0],
        frames_known[0],
        np.where(
            timestamps >= ts_known[-1],
            frames_known[-1],
            np.where(alpha < 0.5, frame_l, frame_r),
        ),
    )

    # Build result
    out = pd.DataFrame(
        {"homography": list(H_interp), "frame_idx": frames},
        index=pd.Index(timestamps, name="timestamp [ns]"),
    )
    return out
