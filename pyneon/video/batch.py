import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video import SceneVideo


def _import_modules():
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "To use parallel processing, the module `joblib` is needed. "
            "Install via: pip install joblib"
        )
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError(
            "To detect AprilTags, the module `pupil-apriltags` is needed. "
            "Install via: pip install pupil-apriltags"
        )


def detect_apriltags_in_batch(
    frames_batch,
    timestamps,
    start_frame_idx,
    tag_family="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
):
    """
    Detect AprilTags on a batch of frames (already read from the video).

    Parameters
    ----------
    frames_batch : list of np.ndarray
        A list of BGR frames to process.
    timestamps : list of int
        The corresponding timestamps for each frame in the batch.
    start_frame_idx : int
        The index of the first frame in the *original* video sequence. We'll use
        this to keep track of 'orig_frame_idx'.
    tag_family : str
        AprilTag family (default 'tag36h11').
    nthreads : int
        Number of CPU threads to let the detector use for each worker.
    quad_decimate : float
        Downsample factor for speed/accuracy tradeoff.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'timestamp [ns]'
        - 'orig_frame_idx'
        - 'tag_id'
        - 'corners'
        - 'center'
    """
    _import_modules()

    # Create a detector (in each worker)
    detector = Detector(
        families=tag_family, nthreads=nthreads, quad_decimate=quad_decimate
    )

    all_detections = []

    for i, frame in enumerate(frames_batch):
        frame_idx = start_frame_idx + i  # actual frame index in the full video
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray_frame)

        for detection in detections:
            corners = detection.corners
            center = np.mean(corners, axis=0)
            all_detections.append(
                {
                    "timestamp [ns]": timestamps[i],  # match to frames_batch
                    "frame_idx": frame_idx,
                    "tag_id": detection.tag_id,
                    "corners": corners,
                    "center": center,
                }
            )

    df = pd.DataFrame(all_detections)
    return df


def detect_apriltags_parallel(
    video,
    tag_family="tag36h11",
    n_jobs=-1,
    nthreads_per_worker=1,
    quad_decimate=1.0,
    chunk_size=500,
) -> pd.DataFrame:
    """
    Parallel AprilTag detection by batching video frames across multiple CPU processes,
    with a tqdm progress bar to track overall progress.

    Parameters
    ----------
    video : OpenCV-like or custom object
        Must allow .read() -> (ret, frame) and have .ts or a known way to get timestamps.
        The length of video.ts is used for the total frame count.
    tag_family : str, optional
        AprilTag family (default 'tag36h11').
    n_jobs : int, optional
        Number of parallel workers (default -1 means 'all cores').
    nthreads_per_worker : int, optional
        Number of CPU threads for each worker's pupil_apriltags Detector.
        If you're spawning multiple processes, it may be wise to keep this small (1 or 2).
    quad_decimate : float, optional
        Downsampling factor for detection (default 1.0).
        Larger => faster detection, but can miss small tags.
    chunk_size : int, optional
        Number of frames to read and process per chunk (default 500).
        Adjust for memory usage vs. efficiency.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'timestamp [ns]'
        - 'orig_frame_idx'
        - 'tag_id'
        - 'corners'
        - 'center'

        If empty, no tags were detected or no frames were read.
    """
    _import_modules()

    total_frames = len(video.ts)  # total number of frames in the video
    all_dfs = []
    frame_index = 0  # which frame we're on

    with tqdm(total=total_frames, desc="Detecting AprilTags (parallel)") as pbar:
        while frame_index < total_frames:
            end_index = min(frame_index + chunk_size, total_frames)
            frames_batch = []
            timestamps_batch = []

            # 1) Read up to `chunk_size` frames
            for idx in range(frame_index, end_index):
                ret, frame = video.read()
                if not ret:
                    # no more frames could be read
                    break
                frames_batch.append(frame)
                timestamps_batch.append(video.ts[idx])

            actual_batch_size = len(frames_batch)
            if actual_batch_size == 0:
                break

            # 2) Split frames_batch among sub-batches for n_jobs
            #    so each parallel worker gets a slice of this chunk.
            if n_jobs <= 0:
                import multiprocessing

                n_jobs_used = multiprocessing.cpu_count()
            else:
                n_jobs_used = n_jobs

            sub_batches = []
            sub_batch_size = max(1, actual_batch_size // n_jobs_used)
            start_sub = 0
            while start_sub < actual_batch_size:
                end_sub = min(start_sub + sub_batch_size, actual_batch_size)
                sub_frames = frames_batch[start_sub:end_sub]
                sub_times = timestamps_batch[start_sub:end_sub]
                sub_info = (
                    sub_frames,
                    sub_times,
                    frame_index + start_sub,  # absolute start idx in full video
                )
                sub_batches.append(sub_info)
                start_sub = end_sub

            # 3) Process sub-batches in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(detect_apriltags_in_batch)(
                    frames_batch=sub_b[0],
                    timestamps=sub_b[1],
                    start_frame_idx=sub_b[2],
                    tag_family=tag_family,
                    nthreads=nthreads_per_worker,
                    quad_decimate=quad_decimate,
                )
                for sub_b in sub_batches
            )

            chunk_df = pd.concat(results, ignore_index=True)
            all_dfs.append(chunk_df)

            # 4) Update progress bar by however many frames we just processed
            pbar.update(actual_batch_size)

            # Advance to next chunk
            frame_index = end_index

    # Combine all partial results
    if not all_dfs:
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)

    # If not empty, set timestamp [ns] as index
    if not final_df.empty:
        final_df.set_index("timestamp [ns]", inplace=True)

    return final_df


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
    if H is None:
        return points.copy()

    points_h = np.column_stack([points, np.ones(len(points))])
    transformed_h = (H @ points_h.T).T
    # Convert from homogeneous to normal 2D, avoiding divide-by-zero
    transformed_2d = transformed_h[:, :2] / np.maximum(transformed_h[:, 2:], 1e-9)
    return transformed_2d


def _compute_homography_for_frame(
    frame: int, detection_df: pd.DataFrame, marker_dict: dict
) -> tuple[int, np.ndarray]:
    """
    Worker function to compute the homography for a single frame.
    Returns (frame, H).

    Parameters
    ----------
    frame : int
        The frame index for which to compute the homography.
    detection_df : pd.DataFrame
        Subset of detections for this frame (only).
    marker_dict : dict
        {marker_id: (4x2) reference corners in target coords}.

    Returns
    -------
    (frame, H) : (int, np.ndarray or None)
        The frame index and the computed homography (3x3) or None.
    """
    if detection_df.empty:
        return (frame, None)

    world_points = []
    screen_points = []

    for _, detection in detection_df.iterrows():
        tag_id = detection["tag_id"]
        if tag_id not in marker_dict:
            # No reference corners for this tag
            continue

        corners_detected = np.array(detection["corners"], dtype=np.float32)
        ref_corners = marker_dict[tag_id]

        # Optional shape check:
        if corners_detected.shape != (4, 2) or ref_corners.shape != (4, 2):
            continue

        world_points.extend(corners_detected)  # add 4 detected corners
        screen_points.extend(ref_corners)  # add 4 reference corners

    world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
    screen_points = np.array(screen_points, dtype=np.float32).reshape(-1, 2)

    if len(world_points) < 4:
        return (frame, None)

    H, _ = cv2.findHomography(world_points, screen_points, cv2.RANSAC, 5.0)
    return (frame, H)


def gaze_to_screen_parallel(
    video,
    detection_df: pd.DataFrame,
    marker_info: pd.DataFrame,
    gaze_df: pd.DataFrame,
    frame_size: tuple[int, int],
    coordinate_system: str = "opencv",
    undistort: bool = True,
    n_jobs: int = 1,
    chunk_size: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """
    Parallelized pipeline that uses all available AprilTags in each frame to compute
    one homography (in parallel), then applies that homography to gaze data for the same frame.

    Parameters
    ----------
    video : SceneVideo-like
        An object with .camera_matrix, .dist_coeffs, etc.
    detection_df : pd.DataFrame
        Must contain columns:
        - 'frame_idx': int
        - 'tag_id': int
        - 'corners': np.ndarray of shape (4, 2)
    marker_info : pd.DataFrame
        Must contain columns:
        - 'marker_id': int (or 'tag_id')
        - 'marker_corners': np.ndarray of shape (4, 2)
    gaze_df : pd.DataFrame
        Must contain columns:
        - 'frame_idx': int
        - 'x', 'y': raw gaze points (in psychoPy or OpenCV coords)
    frame_size : (width, height)
        Pixel resolution of frames, if you need to convert from psychoPy to OpenCV.
    coordinate_system : str, optional
        One of {"opencv", "psychopy"}. If "psychopy", corners + gaze are converted to OpenCV space.
    undistort : bool, optional
        Whether to undistort corners (and optionally gaze) using video.camera_matrix, dist_coeffs.
        By default True.
    n_jobs : int, optional
        Number of processes for parallel homography computation. Default is 1 (serial).

    Returns
    -------
    gaze_df_out : pd.DataFrame
        A copy of `gaze_df` with two new columns: 'x_trans', 'y_trans'.
    homography_for_frame : dict
        A dict mapping frame_idx -> 3x3 homography (or None).
    """
    import warnings
    import math

    _import_modules()
    # --------------------------------------
    # 1) If needed: Convert psychoPy coords -> OpenCV
    # --------------------------------------
    if coordinate_system.lower() == "psychopy":

        def psychopy_coords_to_opencv(coords, frame_size):
            w, h = frame_size
            coords = np.array(coords, dtype=np.float32)
            x_opencv = coords[:, 0] + (w / 2)
            y_opencv = (h / 2) - coords[:, 1]
            return np.column_stack((x_opencv, y_opencv))

        # Convert marker_info
        def convert_marker_corners(c):
            arr = psychopy_coords_to_opencv(c, frame_size)
            return arr.tolist()  # keep it consistent

        marker_info["marker_corners"] = marker_info["marker_corners"].apply(
            convert_marker_corners
        )

        # Convert corners in detection_df
        def convert_detection_corners(row):
            coords = row["corners"]
            arr = psychopy_coords_to_opencv(coords, frame_size)
            return arr

        detection_df = detection_df.copy()
        detection_df["corners"] = detection_df.apply(convert_detection_corners, axis=1)

        # Convert gaze points
        gaze_df = gaze_df.copy()
        xy = gaze_df[["gaze x [px]", "gaze y [px]"]].values.astype(np.float32)
        xy_cv = psychopy_coords_to_opencv(xy, frame_size)
        gaze_df["gaze x [px]"] = xy_cv[:, 0]
        gaze_df["gaze y [px]"] = xy_cv[:, 1]
    else:
        # Just copy to avoid mutating the original
        detection_df = detection_df.copy()
        gaze_df = gaze_df.copy()

    # --------------------------------------
    # 2) Undistort corners if requested
    # --------------------------------------
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

        def undistort_detection_corners(row):
            # row["corners"] should be Nx2
            c = np.array(row["corners"], dtype=np.float32)
            return undistort_points(c, camera_matrix, dist_coeffs)

        detection_df["corners"] = detection_df.apply(
            undistort_detection_corners, axis=1
        )

        # If you want to undistort gaze points as well:
        #   (Often for "x,y" on screen you might skip, but here's how if needed.)
        # xy = gaze_df[["x", "y"]].values
        # xy_und = undistort_points(xy, camera_matrix, dist_coeffs)
        # gaze_df["x"] = xy_und[:, 0]
        # gaze_df["y"] = xy_und[:, 1]

    # --------------------------------------
    # 3) Build a dict for quick marker lookup
    # --------------------------------------
    marker_dict = {}
    for _, row in marker_info.iterrows():
        mid = row.get("marker_id", None)
        if mid is None:
            mid = row.get("tag_id", None)
        if mid is None:
            raise ValueError("marker_info requires a column 'marker_id' or 'tag_id'.")

        marker_dict[mid] = np.array(row["marker_corners"], dtype=np.float32)

    # Group detection_df by frame
    grouped = detection_df.groupby("frame_idx")

    # List all frames we need to process
    frames = sorted(detection_df["frame_idx"].unique())
    total_frames = len(frames)

    # We'll store homographies here
    homographies = {}

    # ----------------------------------------------------------------
    # 4) Chunking + Parallel or Serial
    # ----------------------------------------------------------------
    num_chunks = math.ceil(total_frames / chunk_size)
    current_index = 0

    with tqdm(total=total_frames, desc="Computing homographies by chunk") as pbar:
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total_frames)
            chunk_frames = frames[start:end]

            # Build tasks
            tasks = []
            for frame in chunk_frames:
                if frame in grouped.groups:
                    sub_df = grouped.get_group(frame)
                else:
                    sub_df = pd.DataFrame()  # no detections
                tasks.append((frame, sub_df))

                # Use joblib Parallel
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_compute_homography_for_frame)(
                        int(frame), sub_df, marker_dict
                    )
                    for (frame, sub_df) in tasks
                )

            # Store results
            for frame, H in results:
                homographies[frame] = H

            # Update progress bar by how many frames we processed in this chunk
            pbar.update(len(chunk_frames))

    # ----------------------------------------------------------------
    # 5) Apply homographies to gaze
    # ----------------------------------------------------------------

    gaze_df["x_trans"] = np.nan
    gaze_df["y_trans"] = np.nan
    unique_gaze_frames = gaze_df["frame_idx"].unique()

    for frame in tqdm(unique_gaze_frames, desc="Applying homography to gaze"):
        H = homographies.get(frame, None)
        if H is None:
            continue
        idx_sel = gaze_df["frame_idx"] == frame
        gaze_points = gaze_df.loc[idx_sel, ["x", "y"]].values
        trans = apply_homography(gaze_points, H)
        gaze_df.loc[idx_sel, "x_trans"] = trans[:, 0]
        gaze_df.loc[idx_sel, "y_trans"] = trans[:, 1]

    return gaze_df, homographies
