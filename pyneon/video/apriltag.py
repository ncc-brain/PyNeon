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
    random_access: bool = True,
) -> pd.DataFrame:
    """
    Detect AprilTags in a video and report their data for every processed frame,
    optionally using random access instead of sequential reading.

    Parameters
    ----------
    video : SceneVideo-like
        A video-like object with:
        - .read() -> returns (ret, frame)
        - .ts -> array/list of timestamps (in nanoseconds), same length as total frames
        - (Optional) .set(cv2.CAP_PROP_POS_FRAMES, frame_idx) for random access
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
    random_access : bool, optional
        If True, jump directly to the frames you need (faster for large skip_frames).
        If False, read frames sequentially and skip in a loop (may be faster for
        small skip_frames or if random access is expensive).

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
    all_detections: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute the camera pose for each frame using AprilTag detections stored in a DataFrame,
    handling arbitrary tag orientations.

    Parameters
    ----------
    video : SceneVideo
        Video object containing camera parameters.
    tag_locations_df : pd.DataFrame
        DataFrame containing AprilTag 3D positions, normals, and sizes with columns:
        - 'tag_id': int, ID of the tag
        - 'x', 'y', 'z': float, coordinates of the tag's center
        - 'normal_x', 'normal_y', 'normal_z': float, components of the tag's normal vector
        - 'size': float, side length of the tag in meters
    all_detections : pd.DataFrame
        DataFrame containing AprilTag detections with columns:
        - 'frame_idx': int, frame number
        - 'tag_id': int, ID of the detected tag
        - 'corners': np.ndarray of shape (4, 2), pixel coordinates of the tag's corners
        - 'center': np.ndarray of shape (1, 2), pixel coordinates of the tag's center (optional)

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per frame containing:
        - 'frame_idx': int
        - 'translation_vector': np.ndarray of shape (3,)
        - 'rotation_vector': np.ndarray of shape (3,)
        - 'camera_pos': np.ndarray of shape (3,)
    """

    if all_detections is None or all_detections.empty:
        all_detections = detect_apriltags(video)
        if all_detections.empty:
            print("No AprilTag detections found in the video.")
            return pd.DataFrame(
                columns=[
                    "frame_idx",
                    "translation_vector",
                    "rotation_vector",
                    "camera_pos",
                ]
            )

    camera_matrix = video.camera_matrix
    dist_coeffs = video.dist_coeffs
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    results = []

    # Precompute tag_info dictionary for quick lookup
    # {tag_id: (center_3d, normal, half_size)}
    tag_info_dict = {}
    for _, row in tag_locations_df.iterrows():
        tag_id = row["tag_id"]
        center_3d = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
        normal = np.array(
            [row["normal_x"], row["normal_y"], row["normal_z"]], dtype=np.float32
        )
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
        half_size = row["size"] / 2.0
        tag_info_dict[tag_id] = (center_3d, normal, half_size)

    for frame in all_detections["frame_idx"].unique():
        frame_detections = all_detections.loc[all_detections["frame_idx"] == frame]
        if frame_detections.empty:
            continue

        object_points = []
        image_points = []

        for _, det in frame_detections.iterrows():
            tag_id = det["tag_id"]
            if tag_id not in tag_info_dict:
                continue

            corners_2d = det["corners"]
            center_3d, normal, half_size = tag_info_dict[tag_id]

            # Construct a local coordinate system aligned with the tag's plane
            # Z-axis: normal
            Z = normal

            # Choose a reference vector to avoid degeneracies:
            reference_up = np.array([0, 0, 1], dtype=np.float32)
            # If normal is parallel or close to parallel with reference_up, choose a different reference
            if np.allclose(Z, reference_up, atol=1e-6) or np.allclose(
                Z, -reference_up, atol=1e-6
            ):
                reference_up = np.array([1, 0, 0], dtype=np.float32)

            # X-axis: perpendicular to Z and reference_up
            X = np.cross(Z, reference_up)
            X_norm = np.linalg.norm(X)
            if X_norm < 1e-9:
                # If X is degenerate, pick another reference axis
                reference_up = np.array([0, 1, 0], dtype=np.float32)
                X = np.cross(Z, reference_up)

            X = X / np.linalg.norm(X)
            # Y-axis: perpendicular to Z and X
            Y = np.cross(Z, X)

            # Compute the 3D corners of the tag based on the orientation:
            # We'll arrange corners in a consistent order:
            # bottom-left, bottom-right, top-right, top-left (assuming Z normal facing 'up')
            # Adjust the sign pattern as needed:
            tag_3d_corners = np.array(
                [
                    center_3d + (-half_size) * X + (-half_size) * Y,
                    center_3d + (half_size) * X + (-half_size) * Y,
                    center_3d + (half_size) * X + (half_size) * Y,
                    center_3d + (-half_size) * X + (half_size) * Y,
                ],
                dtype=np.float32,
            )

            object_points.extend(tag_3d_corners)
            image_points.extend(corners_2d)

        if len(object_points) < 4:
            # Not enough points to solve PnP
            continue

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )

        if not success:
            continue

        R, _ = cv2.Rodrigues(rotation_vector)
        camera_pos = -R.T @ translation_vector

        results.append(
            {
                "frame_idx": frame,
                "translation_vector": translation_vector.reshape(-1),
                "rotation_vector": rotation_vector.reshape(-1),
                "camera_pos": camera_pos.reshape(-1),
            }
        )

    return pd.DataFrame(
        results,
        columns=["frame_idx", "translation_vector", "rotation_vector", "camera_pos"],
    )


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform 2D points by a 3x3 homography.

    Parameters
    ----------
    points : np.ndarray of shape (N, 2)
        2D points to be transformed.
    H : np.ndarray of shape (3, 3)
        Homography matrix.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Transformed 2D points.
    """
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed_h = (H @ points_h.T).T
    # Convert from homogeneous to normal 2D
    transformed_2d = transformed_h[:, :2] / transformed_h[:, 2:]
    return transformed_2d


def find_homographies(
    video,
    detection_df: pd.DataFrame,
    marker_info: pd.DataFrame,
    frame_size: tuple[int, int],
    coordinate_system: str = "opencv",
    skip_frames: int = 1,
    undistort: bool = True,
    settings: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute a homography for each frame using available AprilTag detections.

    This function identifies all markers detected in a given frame, looks up their
    "ideal" (reference) positions from `marker_info`, and calls OpenCV's
    `cv2.findHomography` to compute a 3x3 transformation matrix mapping from
    detected corners in the video image to the reference plane (e.g., screen coordinates).

    If the coordinate system is "psychopy", corners in both `marker_info` and
    `detection_df` are first converted to an OpenCV-like pixel coordinate system.
    If `undistort=True` and camera intrinsics are available in the `video` object,
    the marker corners are also undistorted.

    The optional `homography_settings` dictionary allows customizing parameters like
    RANSAC thresholds and maximum iterations. The default is an OpenCV RANSAC method
    with moderate thresholds.

    Parameters
    ----------
    video : SceneVideo-like
        An object containing camera intrinsics (camera_matrix, dist_coeffs) and possibly timestamps.
        If `undistort=True`, these intrinsics are used to undistort marker corners.
    detection_df : pd.DataFrame
        Must contain:
        - 'frame_idx': int
        - 'tag_id': int
        - 'corners': np.ndarray of shape (4, 2) in video or PsychoPy coordinates
    marker_info : pd.DataFrame
        Must contain:
        - 'marker_id' (or 'tag_id'): int
        - 'marker_corners': np.ndarray of shape (4, 2) giving the reference positions
            for each corner (e.g., on a screen plane)
    frame_size : (width, height)
        The pixel resolution of the video frames. Used if `coordinate_system="psychopy"`
        to convert from PsychoPy to OpenCV-style coordinates.
    coordinate_system : str, optional
        One of {"opencv", "psychopy"}. If "psychopy", corners in `detection_df` and
        `marker_info` are converted to OpenCV pixel coords before the homography is computed.
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

        # Convert the reference corners in marker_info
        def convert_marker_corners(c):
            return psychopy_coords_to_opencv(c, frame_size)

        marker_info["marker_corners"] = marker_info["marker_corners"].apply(
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
    for _, row in marker_info.iterrows():
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
    frame_idx_to_ts = dict(zip(
        range(len(video.ts)),
        video.ts
    ))

    records = [
        {
            "timestamp [ns]": frame_idx_to_ts[frame],
            "frame_idx": frame,
            "homography": H
        }
        for frame, H in homography_for_frame.items()
        if frame in frame_idx_to_ts
    ]

    df = pd.DataFrame.from_records(records)
    df = df.set_index("timestamp [ns]")

    return df


def transform_gaze_to_screen(
    gaze_df: pd.DataFrame, homography_for_frame: dict
) -> pd.DataFrame:
    """
    Apply per-frame homographies to gaze points to transform them into a new coordinate system.

    Parameters
    ----------
    gaze_df : pd.DataFrame
        DataFrame containing gaze points with columns:
        - 'frame_idx': int, the frame index
        - 'x', 'y': float, the gaze coordinates in the original coordinate system.
    homography_for_frame : dict
        A dictionary mapping frame indices to 3x3 homography matrices.

    Returns
    -------
    pd.DataFrame
        A copy of `gaze_df` with additional columns:
        - 'x_trans', 'y_trans': the transformed gaze coordinates.
    """
    gaze_df = gaze_df.copy()
    gaze_df["x_trans"] = np.nan
    gaze_df["y_trans"] = np.nan

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
        gaze_trans = apply_homography(gaze_points, H)
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
    dict[int, np.ndarray]
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
    for i in range(len(known_frames) - 1):
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
