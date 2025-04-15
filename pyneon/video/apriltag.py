import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video import NeonVideo


def detect_apriltags(video: "NeonVideo", tag_family: str = "tag36h11",
                    nthreads: int = 4, quad_decimate: float = 1.0,
                    skip_frames: int = 1) -> pd.DataFrame:
    """
    Detect AprilTags in a video and report their data for every frame using pupil_apriltags,
    showing progress with a tqdm progress bar.

    Parameters
    ----------
    video : NeonVideo-like
        A video-like object with:
        - .read() -> returns (ret, frame)
        - .ts -> array/list of timestamps (in nanoseconds), same length as total frames
    tag_family : str, optional
        The AprilTag family to detect (default 'tag36h11').
    nthreads : int, optional
        Number of CPU threads for the detector (default 4).
    quad_decimate : float, optional
        Downsample input frames by this factor for detection (default 1.0).
        Larger values = faster detection, but might miss smaller tags.
    skip_frames : int, optional
        If > 1, skip every N-1 frames (process every skip_frames-th frame).
        This can drastically reduce computation if you only need sparse detection.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing AprilTag detections, with columns:
        - 'timestamp [ns]' (as index): The timestamp of the frame in nanoseconds
        - 'frame_idx': The processed frame number (0-based among the frames you actually read)
        - 'orig_frame_idx': The actual frame index in the video
        - 'tag_id': The ID of the detected AprilTag
        - 'corners': A 4x2 array of the tag corner coordinates (float)
        - 'center': A 1x2 array with the tag center coordinates
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

    # We assume the total number of frames is len(video.ts)
    # If you're using cv2.VideoCapture, you may do something like:
    # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = len(video.ts)

    all_detections = []
    processed_frame_idx = 0

    # Use a for-loop with tqdm for progress
    for actual_frame_idx in tqdm(range(total_frames), desc="Detecting AprilTags"):
        ret, frame = video.read()
        if not ret:
            # No more frames could be read
            break

        # Possibly skip frames
        if actual_frame_idx % skip_frames != 0:
            continue

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = detector.detect(gray_frame)

        for detection in detections:
            corners = detection.corners  # shape (4, 2)
            center = np.mean(corners, axis=0)
            all_detections.append({
                "frame_idx": processed_frame_idx,  # among processed frames
                "orig_frame_idx": actual_frame_idx,  # actual frame index
                "timestamp [ns]": video.ts[actual_frame_idx],
                "tag_id": detection.tag_id,
                "corners": corners,
                "center": center,
            })

        processed_frame_idx += 1

    # Convert to DataFrame
    df = pd.DataFrame(all_detections)
    if df.empty:
        return df  # no detections at all

    # Set timestamp [ns] as index
    df.set_index("timestamp [ns]", inplace=True)
    return df

def estimate_camera_pose(
    video: "NeonVideo",
    tag_locations_df: pd.DataFrame,
    all_detections: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute the camera pose for each frame using AprilTag detections stored in a DataFrame,
    handling arbitrary tag orientations.

    Parameters
    ----------
    video : NeonVideo
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


def psychopy_coords_to_opencv(coords: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """
    Converts coordinates from a PsychoPy-like coordinate system
    to an OpenCV-like (pixel-based) coordinate system.

    PsychoPy's default coordinate system assumptions vary a lot
    depending on your setup. A common assumption might be:
    - (0,0) in the center
    - Y increasing upwards
    - X increasing to the right
    Meanwhile, OpenCV is:
    - (0,0) in the *top-left*
    - Y increasing downwards
    - X increasing to the right

    Parameters
    ----------
    coords : np.ndarray of shape (N, 2)
        The (x, y) coordinates in PsychoPy-like space.
    frame_size : (width, height)
        The pixel dimensions of the frame.

    Returns
    -------
    np.ndarray of shape (N, 2)
        The transformed coordinates in OpenCV space.
    """
    w, h = frame_size
    # PsychoPy center (0,0) => center of screen
    # Convert to top-left as (0,0):
    #   x_opencv = x_psychopy + (w/2)
    #   y_opencv = (h/2) - y_psychopy
    # (Because if y_psychopy is +, that is "up" in PsychoPy, but in OpenCV, + is down.)
    x_opencv = coords[:, 0] + (w / 2)
    y_opencv = (h / 2) - coords[:, 1]
    return np.column_stack((x_opencv, y_opencv))


def compute_2d_homography(corners_2d_src: np.ndarray, corners_2d_dst: np.ndarray) -> np.ndarray:
    """
    Compute a 2D homography (3x3 matrix) mapping corners_2d_src to corners_2d_dst.

    Parameters
    ----------
    corners_2d_src : np.ndarray of shape (N, 2)
        Source pixel coordinates (e.g. detected tag corners).
    corners_2d_dst : np.ndarray of shape (N, 2)
        Destination pixel coordinates (the reference layout or known 2D positions).

    Returns
    -------
    np.ndarray
        A 3x3 homography matrix.
    """
    H, mask = cv2.findHomography(corners_2d_src, corners_2d_dst, cv2.RANSAC, 5.0)
    return H


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


def gaze_to_screen(
    video,
    detection_df: pd.DataFrame,
    marker_info: pd.DataFrame,
    gaze_df: pd.DataFrame,
    frame_size: tuple[int, int],
    coordinate_system: str = "opencv",
    undistort: bool = True,
) -> pd.DataFrame:
    """
    Pipeline that uses all available AprilTags in each frame to compute one homography,
    then applies that homography to gaze data for the same frame.

    Parameters
    ----------
    video : NeonVideo-like
        An object containing camera intrinsics (camera_matrix, dist_coeffs) and possibly timestamps.
    detection_df : pd.DataFrame
        Must contain:
        - 'frame_idx': int
        - 'tag_id': int
        - 'corners': np.ndarray of shape (4, 2) in *video* coordinates (or psychopy coords, see below).
    marker_info : pd.DataFrame
        Must contain:
        - 'marker_id' or 'tag_id': int
        - 'marker_corners': np.ndarray of shape (4, 2) giving the "ideal" or reference positions
        for each corner of this tag in some target coordinate system (e.g., screen coordinates).
        The row for each marker_id should define the "destination" corners for that tag.
    gaze_df : pd.DataFrame
        Must contain:
        - 'frame_idx': int (or you can do a timestamp-based merge_asof first)
        - 'x', 'y': raw gaze points (in either psychopy or opencv coords).
    frame_size : (width, height)
        The pixel resolution of your video frames, used if you need to convert from psychoPy to OpenCV coordinates.
    coordinate_system : str, optional
        One of {"opencv", "psychopy"}. If "psychopy", corners and gaze points will be converted to OpenCV pixel coords.
    undistort : bool, optional
        Whether to undistort corners and gaze points using video.camera_matrix and video.dist_coeffs.

    Returns
    -------
    pd.DataFrame
        A copy of `gaze_df` with additional columns 'x_trans', 'y_trans' that are
        the gaze coordinates transformed by the per-frame homography.
    """

    # -----------------------------------------------------------------
    # 1. Convert from PsychoPy coords to OpenCV if necessary
    # -----------------------------------------------------------------
    if coordinate_system.lower() == "psychopy":
        # Example transform function
        def psychopy_coords_to_opencv(coords: np.ndarray, frame_size):
            w, h = frame_size
            x_opencv = coords[:, 0] + (w / 2)
            y_opencv = (h / 2) - coords[:, 1]
            return np.column_stack((x_opencv, y_opencv))

        # Convert corners in detection_df
        def convert_detection_corners(row):
            c = row["corners"]
            return psychopy_coords_to_opencv(c, frame_size)
        detection_df["corners"] = detection_df.apply(convert_detection_corners, axis=1)

        # Convert gaze
        gaze_array = gaze_df[["x", "y"]].values
        gaze_cv = psychopy_coords_to_opencv(gaze_array, frame_size)
        gaze_df["x"] = gaze_cv[:, 0]
        gaze_df["y"] = gaze_cv[:, 1]

        # Convert the reference corners in marker_info
        def convert_marker_corners(c):
            return psychopy_coords_to_opencv(c, frame_size)
        marker_info["marker_corners"] = marker_info["marker_corners"].apply(convert_marker_corners)

    # -----------------------------------------------------------------
    # 2. Undistort corners & gaze if desired
    # -----------------------------------------------------------------
    camera_matrix = getattr(video, "camera_matrix", None)
    dist_coeffs = getattr(video, "dist_coeffs", None)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    if undistort and camera_matrix is not None:
        def undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
            pts_for_cv = points.reshape((-1, 1, 2)).astype(np.float32)
            undist = cv2.undistortPoints(pts_for_cv, K, D)
            # undistortPoints outputs normalized coords => multiply back by K
            ones = np.ones((undist.shape[0], 1, 1), dtype=np.float32)
            undist_hom = np.concatenate([undist, ones], axis=2)
            pixel = np.einsum('ij,nkj->nki', K, undist_hom)
            pixel = pixel[:, 0, :2] / pixel[:, 0, 2:]
            return pixel

        # Undistort detection corners
        def undistort_detection_corners(row):
            c = row["corners"]
            return undistort_points(c, camera_matrix, dist_coeffs)
        detection_df["corners"] = detection_df.apply(undistort_detection_corners, axis=1)

    # -----------------------------------------------------------------
    # 3. Compute a homography for each frame using *all* tag detections
    # -----------------------------------------------------------------
    frames = detection_df["frame_idx"].unique()
    homography_for_frame = {}

    for frame in frames:
        frame_detections = detection_df.loc[detection_df["frame_idx"] == frame]
        if frame_detections.empty:
            homography_for_frame[frame] = None
            continue

        world_points = []   # from the camera's perspective (the "detected corners")
        screen_points = []  # from the reference plane or "ideal" positions

        for _, detection in frame_detections.iterrows():
            tag_id = detection["tag_id"]
            # match row in marker_info
            row_marker = marker_info.loc[marker_info["marker_id"] == tag_id]
            if row_marker.empty:
                # no reference corners for that tag
                continue

            corners_detected = detection["corners"]  # shape (4,2)
            ref_corners = row_marker.iloc[0]["marker_corners"]  # shape (4,2)
            if corners_detected.shape[0] != 4 or ref_corners.shape[0] != 4:
                # skip if corner count doesn't match
                continue

            # Append each corner pair (detected, reference)
            world_points.append(corners_detected[0])
            world_points.append(corners_detected[1])
            world_points.append(corners_detected[2])
            world_points.append(corners_detected[3])

            screen_points.append(ref_corners[0])
            screen_points.append(ref_corners[1])
            screen_points.append(ref_corners[2])
            screen_points.append(ref_corners[3])

        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
        screen_points = np.array(screen_points, dtype=np.float32).reshape(-1, 2)

        if len(world_points) < 4:
            # Not enough corners to compute a homography
            homography_for_frame[frame] = None
            continue

        H, mask = cv2.findHomography(world_points, screen_points, cv2.RANSAC, 5.0)
        homography_for_frame[frame] = H

    # -----------------------------------------------------------------
    # 4. Apply that homography to gaze points per frame
    # -----------------------------------------------------------------
    # We'll add "x_trans", "y_trans" columns to the gaze DataFrame
    gaze_df = gaze_df.copy()
    gaze_df["x_trans"] = np.nan
    gaze_df["y_trans"] = np.nan

    for frame in gaze_df["frame_idx"].unique():
        idx_sel = (gaze_df["frame_idx"] == frame)
        H = homography_for_frame.get(frame, None)
        if H is None:
            # no valid homography
            continue
        # transform the gaze coords
        gaze_points = gaze_df.loc[idx_sel, ["x", "y"]].values
        gaze_trans = apply_homography(gaze_points, H)
        gaze_df.loc[idx_sel, "x_trans"] = gaze_trans[:, 0]
        gaze_df.loc[idx_sel, "y_trans"] = gaze_trans[:, 1]

    # Done
    return gaze_df, homography_for_frame