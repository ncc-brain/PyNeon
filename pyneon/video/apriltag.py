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


def detect_apriltags(
    video: "SceneVideo",
    tag_family: str = "tag36h11",
    nthreads: int = 4,
    quad_decimate: float = 1.0,
    skip_frames: int = 1,
    return_diagnostics: bool = False,
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
        Default is 1 (process every frame).
    return_diagnostics : bool, optional
        If True, return additional diagnostic information.
        Default is False.

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
                results = {
                    "processed_frame_idx": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
                    "timestamp [ns]": video.ts[actual_frame_idx],
                    "tag_id": detection.tag_id,
                    "corners": corners,
                    "center": center,
                }

                if return_diagnostics:
                    results["hamming"] = detection.hamming
                    results["decision_margin"] = detection.decision_margin

                all_detections.append(results)

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
                results = {
                    "processed_frame_idx": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
                    "timestamp [ns]": video.ts[actual_frame_idx],
                    "tag_id": detection.tag_id,
                    "corners": corners,
                    "center": center,
                }

                if return_diagnostics:
                    results["hamming"] = detection.hamming
                    results["decision_margin"] = detection.decision_margin

                all_detections.append(results)

            processed_frame_idx += 1

    # -----------------------------------------------------------------------
    # Create and return the DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_detections)
    
    if df.empty:
        return df  # no detections found

    df["method"] = "apriltag"
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

