import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .video import SceneVideo


import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_apriltags(
    video: "SceneVideo",
    tag_family: str = "tag36h11",
    nthreads: int = 4,
    quad_decimate: float = 1.0,
    skip_frames: int = 1,
) -> pd.DataFrame:
    """
    Detect AprilTags in a video and report their data for every processed frame,
    optionally using random access instead of sequential reading.

    Parameters
    ----------
    video : SceneVideo
        Scene video to detect AprilTags from.
    tag_family : str, optional
        The AprilTag family to detect (default 'tag36h11').
    nthreads : int, optional
        Number of CPU threads for the detector (default 4).
    quad_decimate : float, optional
        Downsample input frames by this factor for detection (default 1.0).
        Larger values = faster detection, but might miss smaller tags.
    skip_frames : int, optional
        If > 1, detect tags only in every Nth frame.
        E.g., skip_frames=5 will process frames 0, 5, 10, 15, etc.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing AprilTag detections, with columns:
        - 'timestamp [ns]' (index): The timestamp of the frame
        - 'processed_frame_idx': The count of processed frames (0-based)
        - 'frame_idx': The actual frame index in the video
        - 'tag_id': The ID of the detected AprilTag
        - 'corners': (4,2) array of tag corner coordinates
        - 'center': (1,2) array for the tag center
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError(
            "To detect AprilTags, the module `pupil-apriltags` is needed. "
            "Install via: pip install pupil-apriltags"
        )

    # Initialize the detector
    detector = Detector(
        families=tag_family,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
    )

    random_access = True
    if skip_frames < 1:
        raise ValueError("skip_frames must be >= 1")
    if skip_frames < 5:
        print(
            "Warning: skip_frames < 5 may be inefficient with random access. Switching to sequential access."
        )
        random_access = False  # sequential access is faster for small skips

    total_frames = len(video.ts)
    all_detections = []
    processed_frame_idx = 0  # counts how many frames we've actually processed

    # -----------------------------------------------------------------------
    # Random-Access Approach
    # -----------------------------------------------------------------------
    if random_access:
        frames_to_process = range(0, total_frames, skip_frames)
        for actual_frame_idx in tqdm(
            frames_to_process, desc="Detecting AprilTags (random access)"
        ):
            # Seek directly to the desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
            ret, frame = video.read()
            if not ret:
                break

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect
            detections = detector.detect(gray_frame)

            # Save
            for detection in detections:
                corners = detection.corners
                center = np.mean(corners, axis=0)
                all_detections.append(
                    {
                        "processed_frame_idx": processed_frame_idx,
                        "frame_idx": actual_frame_idx,
                        "timestamp [ns]": video.ts[actual_frame_idx],
                        "tag_id": detection.tag_id,
                        "corners": corners,
                        "center": center,
                    }
                )

            processed_frame_idx += 1

    # -----------------------------------------------------------------------
    # Sequential Approach
    # -----------------------------------------------------------------------
    else:
        # We'll just read frames in order, skipping those we don't want
        for actual_frame_idx in tqdm(
            range(total_frames), desc="Detecting AprilTags (sequential)"
        ):
            ret, frame = video.read()
            if not ret:
                break

            if actual_frame_idx % skip_frames != 0:
                continue  # skip

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect
            detections = detector.detect(gray_frame)

            # Save
            for detection in detections:
                corners = detection.corners
                center = np.mean(corners, axis=0)
                all_detections.append(
                    {
                        "processed_frame_idx": processed_frame_idx,
                        "frame_idx": actual_frame_idx,
                        "timestamp [ns]": video.ts[actual_frame_idx],
                        "tag_id": detection.tag_id,
                        "corners": corners,
                        "center": center,
                    }
                )

            processed_frame_idx += 1

    # -----------------------------------------------------------------------
    # Create and return the DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_detections)
    if df.empty:
        return df  # no detections found

    df.set_index("timestamp [ns]", inplace=True)
    return df


def estimate_camera_pose(
    video: "SceneVideo",
    tag_locations_df: pd.DataFrame,
    all_detections: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Estimate the camera pose for every frame by solving a Perspective-n-Point
    (PnP) problem based on AprilTag detections.

    Parameters
    ----------
    video :
        ``SceneVideo`` instance providing the frames' timestamps and the
        intrinsic matrices ``camera_matrix`` and ``dist_coeffs``.
    tag_locations_df :
        pandas.DataFrame describing the world-coordinates of each AprilTag.
        Required columns::

            "tag_id"   : int
            "pos_vec"  : list[float] length 3, [x, y, z] position of the tag
            "norm_vec" : list[float] length 3, [nx, ny, nz] front-face normal
            "size"     : float, edge length in meters
    all_detections :
        pandas.DataFrame with per-frame tag detections.  If *None* the function
        calls ``detect_apriltags(video)``.  Expected columns::

            "frame_idx" : int
            "tag_id"    : int
            "corners"   : ndarray (4, 2) pixel coordinates (TL, TR, BR, BL)

    Returns
    -------
    pandas.DataFrame
        One row per processed frame with columns::

            "frame_idx"          : int
            "translation_vector" : ndarray (3,)
            "rotation_vector"    : ndarray (3,)
            "camera_pos"         : ndarray (3,) camera position in world coord.
    """

    # ------------------------------------------------------------------ prepare detections
    if all_detections is None or all_detections.empty:
        from .apriltag import detect_apriltags  # local import to avoid cycle

        all_detections = detect_apriltags(video)
        if all_detections.empty:
            return pd.DataFrame(
                columns=[
                    "frame_idx",
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
    for frame in all_detections["frame_idx"].unique():
        det_frame = all_detections.loc[all_detections["frame_idx"] == frame]
        if det_frame.empty:
            continue

        object_pts, image_pts = [], []

        for _, det in det_frame.iterrows():
            corners_2d = np.asarray(det["corners"], dtype=np.float32)

            tid = int(det["tag_id"])
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
                "frame_idx": int(frame),
                "translation_vector": t_vec.reshape(-1),
                "rotation_vector": r_vec.reshape(-1),
                "camera_pos": cam_pos.reshape(-1),
            }
        )

    return pd.DataFrame(
        results,
        columns=["frame_idx", "translation_vector", "rotation_vector", "camera_pos"],
    )


def _apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform 2D points by a 3x3 homography.

    Parameters
    ----------
    points : numpy.ndarray of shape (N, 2)
        2D points to be transformed.
    H : numpy.ndarray of shape (3, 3)
        Homography matrix.

    Returns
    -------
    numpy.ndarray of shape (N, 2)
        Transformed 2D points.
    """
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed_h = (H @ points_h.T).T
    # Convert from homogeneous to normal 2D
    transformed_2d = transformed_h[:, :2] / transformed_h[:, 2:]
    return transformed_2d


def find_homographies(
    video: "SceneVideo",
    detection_df: pd.DataFrame,
    tag_info: pd.DataFrame,
    frame_size: tuple[int, int],
    coordinate_system: str = "opencv",
    skip_frames: int = 1,
    undistort: bool = True,
    settings: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute a homography for each frame using available AprilTag detections.

    This function identifies all markers detected in a given frame, looks up their
    "ideal" (reference) positions from `tag_info`, and calls OpenCV's
    `cv2.findHomography` to compute a 3x3 transformation matrix mapping from
    detected corners in the video image to the reference plane (e.g., screen coordinates).

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
            for each corner (e.g., on a screen plane)
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

    Returns
    -------
    dict
        A dictionary mapping each frame index (`frame_idx`: int) to its corresponding
        homography matrix (3x3 NumPy array) or None if insufficient markers or points
        were available to compute a valid homography.
    """

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
        screen_points = []  # from the reference plane or "ideal" positions

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
            screen_points.extend(ref_corners)  # add 4 reference coords

        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
        screen_points = np.array(screen_points, dtype=np.float32).reshape(-1, 2)

        if len(world_points) < 4:
            # Not enough corners to compute a homography
            homography_for_frame[frame] = None
            continue

        H, mask = cv2.findHomography(world_points, screen_points, **default_settings)
        homography_for_frame[frame] = H

    if skip_frames != 1:
        # Upsample the homographies to fill in skipped frames
        max_frame = max(frames)
        homography_for_frame = _upsample_homographies(homography_for_frame, max_frame)

    # Get timestamps for each frame_idx
    frame_idx_to_ts = dict(zip(range(len(video.ts)), video.ts))

    records = [
        {"timestamp [ns]": frame_idx_to_ts[frame], "frame_idx": frame, "homography": H}
        for frame, H in homography_for_frame.items()
        if frame in frame_idx_to_ts
    ]

    df = pd.DataFrame.from_records(records)
    df = df.set_index("timestamp [ns]")

    return df


def transform_gaze_to_screen(
    gaze_df: pd.DataFrame, homographies: pd.DataFrame
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

    # convert homographies to dict
    homography_for_frame = {
        int(row["frame_idx"]): row["homography"] for _, row in homographies.iterrows()
    }

    for frame in tqdm(
        gaze_df["frame_idx"].unique(), desc="Applying homography to gaze points"
    ):
        idx_sel = gaze_df["frame_idx"] == frame
        H = homography_for_frame.get(frame, None)
        if H is None:
            # no valid homography
            continue
        # transform the gaze coords
        gaze_points = gaze_df.loc[idx_sel, ["gaze x [px]", "gaze y [px]"]].values
        gaze_trans = _apply_homography(gaze_points, H)
        gaze_df.loc[idx_sel, "x_trans"] = gaze_trans[:, 0]
        gaze_df.loc[idx_sel, "y_trans"] = gaze_trans[:, 1]

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
