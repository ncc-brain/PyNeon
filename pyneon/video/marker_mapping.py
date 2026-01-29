from typing import TYPE_CHECKING, Literal, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils.doc_decorators import fill_doc
from .utils import marker_family_to_dict

if TYPE_CHECKING:
    from .video import Video

DETECTED_MARKERS_COLUMNS = {
    "frame index",
    "marker family",
    "marker id",
    "marker name",
    "top left x [px]",
    "top left y [px]",
    "top right x [px]",
    "top right y [px]",
    "bottom right x [px]",
    "bottom right y [px]",
    "bottom left x [px]",
    "bottom left y [px]",
    "center x [px]",
    "center y [px]",
}

MARKERS_LAYOUT_COLUMNS = {
    "marker name",
    "size",
    "center x",
    "center y",
}


@fill_doc
def detect_markers(
    video: "Video",
    marker_family: str | list[str],
    step: int = 1,
    detection_window: Optional[tuple[int | float, int | float]] = None,
    detection_window_unit: Literal["frame", "time", "timestamp"] = "frame",
    detector_parameters: Optional[cv2.aruco.DetectorParameters] = None,
) -> Stream:
    """
    Detect fiducial markers (AprilTag or ArUco) in a video and report their data for every processed frame.
    Uses random access to read frames for dense sampling (step < 5), otherwise sequential streaming.

    Parameters
    ----------
    video : Video
        Scene video to detect markers from.
    %(detect_markers_params)s
    %(detect_markers_return)s
    """
    # Normalize marker family input to a list and create detectors for each
    families: list[str] = (
        marker_family if isinstance(marker_family, list) else [marker_family]
    )

    # Use provided detector_parameters or create a default instance
    if detector_parameters is None:
        detector_parameters = cv2.aruco.DetectorParameters()

    detectors: list[tuple[str, str, cv2.aruco.ArucoDetector]] = []
    for fam in families:
        fam_type, aruco_dict = marker_family_to_dict(fam)
        detectors.append(
            (fam, fam_type, cv2.aruco.ArucoDetector(aruco_dict, detector_parameters))
        )

    if step < 1:
        raise ValueError("step must be >= 1")

    # Specify indices of frames to process
    if detection_window is None:  # full video
        start_frame_idx = 0
        end_frame_idx = len(video.ts) - 1
    else:
        start, end = detection_window
        if detection_window_unit == "frame":
            # Already in frame indices, use directly
            start_frame_idx, end_frame_idx = int(start), int(end)
        elif detection_window_unit == "time":
            start_frame_idx = int(np.searchsorted(video.times, start, side="left"))
            end_frame_idx = int(np.searchsorted(video.times, end, side="right")) - 1
        elif detection_window_unit == "timestamp":
            # Convert from nanosecond timestamps to frame indices
            start_frame_idx = int(np.searchsorted(video.ts, start, side="left"))
            end_frame_idx = int(np.searchsorted(video.ts, end, side="right")) - 1

    def _process_frame(frame_idx: int, gray_frame: np.ndarray) -> list[dict]:
        """Run detection on a single grayscale frame across all detectors."""
        records: list[dict] = []
        for fam_name, fam_type, det in detectors:
            all_corners, all_ids, _ = det.detectMarkers(gray_frame)
            if all_ids is None:
                continue

            for corners, marker_id in zip(all_corners, all_ids):
                corners = corners.reshape((4, 2))
                if fam_type == "april":
                    # For AprilTags, corners start with bottom right
                    # For ArUco, corners start with top left
                    # See https://stackoverflow.com/questions/79044142
                    corners = corners[[2, 3, 0, 1], :]
                center = np.mean(corners, axis=0)
                records.append(
                    {
                        "timestamp [ns]": video.ts[frame_idx],
                        "frame index": frame_idx,
                        "marker family": fam_name,
                        "marker id": int(marker_id[0]),
                        "marker name": f"{fam_name}_{marker_id[0]}",
                        "top left x [px]": corners[0, 0],
                        "top left y [px]": corners[0, 1],
                        "top right x [px]": corners[1, 0],
                        "top right y [px]": corners[1, 1],
                        "bottom right x [px]": corners[2, 0],
                        "bottom right y [px]": corners[2, 1],
                        "bottom left x [px]": corners[3, 0],
                        "bottom left y [px]": corners[3, 1],
                        "center x [px]": center[0],
                        "center y [px]": center[1],
                    }
                )
        return records

    detected_markers = []
    frames_to_process = list(range(start_frame_idx, end_frame_idx + 1, step))

    use_random_access = step < 5

    if use_random_access:
        for actual_frame_idx in tqdm(frames_to_process, desc="Detecting markers"):
            gray_frame = video.read_gray_frame_at(actual_frame_idx)
            if gray_frame is None:
                break
            records = _process_frame(actual_frame_idx, gray_frame)
            detected_markers.extend(records)
    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        for actual_frame_idx in tqdm(
            range(start_frame_idx, end_frame_idx + 1),
            desc="Detecting markers",
        ):
            ret, frame = video.read()
            if not ret:
                break
            if actual_frame_idx % step != 0:
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            records = _process_frame(actual_frame_idx, gray_frame)
            detected_markers.extend(records)

    df = pd.DataFrame(detected_markers)
    if df.empty:
        raise ValueError("No marker detected.")

    df.set_index("timestamp [ns]", inplace=True)
    return Stream(df)


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
    detected_markers: Stream,
    marker_layout: pd.DataFrame,
    valid_markers: int = 2,
    settings: dict = {},
) -> Stream:
    """
    Compute a homography for each frame using available marker detections.

    This function identifies all markers detected in a given frame, looks up their
    reference positions from `marker_layout`, and calls OpenCV's `cv2.findHomography`
    to compute a 3x3 transformation matrix mapping from detected corners in the video
    image to the reference plane (surface coordinates). The homography matrices are
    flattened into 9 columns for convenient storage and processing.

    Parameters
    ----------
    detected_markers : Stream
        Stream containing per-frame marker detections as returned by
        :meth:`detect_markers`.
    marker_layout : pandas.DataFrame
        DataFrame with the following columns:
            - 'marker name': full marker identifier (family + id, e.g., 'tag36h11_1')
            - 'size': size of the marker in the reference plane units
            - 'center x': x center of the marker in OpenCV coordinates
            - 'center y': y center of the marker in OpenCV coordinates
    valid_markers : int, optional
        Minimum number of markers required to compute a homography. Defaults to 2.
    settings : dict, optional
        A dictionary of parameters passed to `cv2.findHomography`. For example:
        {
            "method": cv2.RANSAC,
            "ransacReprojThreshold": 2.0,
            "maxIters": 500,
            "confidence": 0.98,
        }
        Defaults to cv2.LMEDS method.

    Returns
    -------
    Stream
        A Stream indexed by 'timestamp [ns]' with columns
        'homography (0,0)' through 'homography (2,2)': The 9 elements of the
        flattened 3x3 homography matrix
    """
    detection_df = detected_markers.data
    if not DETECTED_MARKERS_COLUMNS.issubset(detection_df.columns):
        raise ValueError(
            f"detected_markers.data must contain the following columns: {', '.join(DETECTED_MARKERS_COLUMNS)}"
        )

    # Check if marker_layout has required columns
    if not MARKERS_LAYOUT_COLUMNS.issubset(marker_layout.columns):
        raise ValueError(
            f"marker_layout must contain the following columns: {', '.join(MARKERS_LAYOUT_COLUMNS)}"
        )

    # Construct (4, 2) corner arrays in surface coordinates
    marker_layout = marker_layout.copy()
    marker_layout["marker_corners"] = marker_layout.apply(
        lambda row: np.array(
            [
                [
                    row["center x"] - row["size"] / 2,
                    row["center y"] - row["size"] / 2,
                ],  # top left
                [
                    row["center x"] + row["size"] / 2,
                    row["center y"] - row["size"] / 2,
                ],  # top right
                [
                    row["center x"] + row["size"] / 2,
                    row["center y"] + row["size"] / 2,
                ],  # bottom right
                [
                    row["center x"] - row["size"] / 2,
                    row["center y"] + row["size"] / 2,
                ],  # bottom left
            ],
            dtype=np.float32,
        ),
        axis=1,
    )

    # if undistort:
    #     def undistort_points(
    #         points: np.ndarray, K: np.ndarray, D: np.ndarray
    #     ) -> np.ndarray:
    #         pts_for_cv = points.reshape((-1, 1, 2)).astype(np.float32)
    #         undist = cv2.undistortPoints(pts_for_cv, K, D)
    #         # undistortPoints outputs normalized coords => multiply back by K
    #         ones = np.ones((undist.shape[0], 1, 1), dtype=np.float32)
    #         undist_hom = np.concatenate([undist, ones], axis=2)
    #         pixel = np.einsum("ij,nkj->nki", K, undist_hom)
    #         pixel = pixel[:, 0, :2] / pixel[:, 0, 2:]
    #         return pixel

    #     # Undistort detection corners
    #     def undistort_detection_corners(row):
    #         c = np.array(
    #             [
    #                 [row["corner 0 x [px]"], row["corner 0 y [px]"]],
    #                 [row["corner 1 x [px]"], row["corner 1 y [px]"]],
    #                 [row["corner 2 x [px]"], row["corner 2 y [px]"]],
    #                 [row["corner 3 x [px]"], row["corner 3 y [px]"]],
    #             ]
    #         )
    #         return undistort_points(c, video.camera_matrix, video.distortion_coefficients)

    #     # Reconstruct corners array and store back
    #     undistorted_corners = detection_df.apply(undistort_detection_corners, axis=1)
    #     for i in range(4):
    #         detection_df[f"corner {i} x [px]"] = undistorted_corners.apply(
    #             lambda c: c[i, 0]
    #         )
    #         detection_df[f"corner {i} y [px]"] = undistorted_corners.apply(
    #             lambda c: c[i, 1]
    #         )

    # compute homography for each frame using all marker detections
    default_settings = {
        "method": cv2.LMEDS  # Disable RANSAC completely
    }
    default_settings.update(settings)

    unique_timestamps = detection_df.index.unique()
    homography_for_frame = {}

    for ts in tqdm(unique_timestamps, desc="Computing homographies"):
        frame_detections = detection_df.loc[ts]

        world_points = []  # from the camera's perspective (detected corners)
        surface_points = []  # from the reference plane or "ideal" positions

        for _, detection in frame_detections.iterrows():
            marker_name = detection["marker name"]
            if marker_name not in marker_layout["marker name"].values:
                continue

            # Reconstruct corners array from individual columns
            corners_detected = np.array(
                [
                    [detection["top left x [px]"], detection["top left y [px]"]],
                    [detection["top right x [px]"], detection["top right y [px]"]],
                    [
                        detection["bottom right x [px]"],
                        detection["bottom right y [px]"],
                    ],
                    [detection["bottom left x [px]"], detection["bottom left y [px]"]],
                ],
                dtype=np.float32,
            )
            ref_corners = marker_layout.loc[
                marker_layout["marker name"] == marker_name, "marker_corners"
            ].values[0]

            # Shape check:
            if corners_detected.shape != (4, 2) or ref_corners.shape != (4, 2):
                raise ValueError(
                    f"Marker corners must have shape (4, 2), got {corners_detected.shape} and {ref_corners.shape}"
                )

            # Extend our list of corner correspondences
            world_points.extend(corners_detected)  # add 4 corner coords
            surface_points.extend(ref_corners)  # add 4 reference coords

        world_points = np.array(world_points, dtype=np.float32).reshape(-1, 2)
        surface_points = np.array(surface_points, dtype=np.float32).reshape(-1, 2)

        if len(world_points) < valid_markers * 4:
            # Not enough corners to compute a homography
            continue

        homography, _ = cv2.findHomography(
            world_points, surface_points, **default_settings
        )
        homography_for_frame[ts] = homography

    # Reorganize homographies into DataFrame with 9 columns (flattened 3x3 matrix)
    records = []
    for ts, homography in homography_for_frame.items():
        record = {
            "timestamp [ns]": ts,
        }
        # Flatten 3x3 homography matrix into 9 columns
        if homography is not None:
            for i in range(3):
                for j in range(3):
                    record[f"homography ({i},{j})"] = homography[i, j]
            records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No homographies could be computed from the detections.")

    return Stream(df)


def gaze_on_surface(gaze_df: pd.DataFrame, homographies: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-frame or per-sample homographies to gaze points.

    If `gaze_df` and `homographies` have the same index, homographies are applied
    point-wise (efficiently vectorized). Otherwise, they are applied by grouping
    `gaze_df` by 'frame index'.

    Parameters
    ----------
    gaze_df : pandas.DataFrame
        DataFrame containing gaze points with columns 'gaze x [px]' and 'gaze y [px]'.
    homographies : pandas.DataFrame
        DataFrame containing homography matrices in a 'homography' column.

    Returns
    -------
    pandas.DataFrame
        A copy of `gaze_df` with additional columns 'gaze x [surface coords]' and 'gaze y [surface coords]'.
    """
    gaze_df = gaze_df.copy()
    gaze_df["gaze x [surface coord]"] = np.nan
    gaze_df["gaze y [surface coord]"] = np.nan

    # Case 1: DataFrames are aligned by index (e.g. gaze-sampled homographies)
    if gaze_df.index.equals(homographies.index):
        # Filter for rows where both have data
        valid_mask = (
            gaze_df[["gaze x [px]", "gaze y [px]"]].notna().all(axis=1)
            & homographies["homography"].notna()
        )

        if not valid_mask.any():
            return gaze_df

        idx_valid = gaze_df.index[valid_mask]
        points = gaze_df.loc[idx_valid, ["gaze x [px]", "gaze y [px]"]].values
        # Stack 3x3 matrices into (N, 3, 3)
        H_stack = np.stack(homographies.loc[idx_valid, "homography"].values)

        # Vectorized homography application: [x', y', w']^T = H @ [x, y, 1]^T
        points_h = np.column_stack([points, np.ones(len(points))])
        transformed_h = np.einsum("nij,nj->ni", H_stack, points_h)

        # Convert from homogeneous to normal 2D
        transformed_2d = transformed_h[:, :2] / transformed_h[:, 2:]
        gaze_df.loc[idx_valid, "gaze x [surface coord]"] = transformed_2d[:, 0]
        gaze_df.loc[idx_valid, "gaze y [surface coord]"] = transformed_2d[:, 1]
        return gaze_df

    # convert homographies to dict
    homography_for_frame = {
        int(row["frame index"]): row["homography"] for _, row in homographies.iterrows()
    }

    for frame in tqdm(
        gaze_df["frame index"].unique(), desc="Applying homography to gaze points"
    ):
        idx_sel = gaze_df["frame index"] == frame
        H = homography_for_frame.get(frame, None)
        if H is None:
            # no valid homography
            continue
        # transform the gaze coords
        gaze_points = gaze_df.loc[idx_sel, ["gaze x [px]", "gaze y [px]"]].values
        gaze_trans = _apply_homography(gaze_points, H)
        gaze_df.loc[idx_sel, "gaze x [surface coord]"] = gaze_trans[:, 0]
        gaze_df.loc[idx_sel, "gaze y [surface coord]"] = gaze_trans[:, 1]

    return gaze_df


def _upsample_homographies(
    homographies_dict: dict[int | np.int64, np.ndarray],
    max_frame: int | np.int64,
    max_gap: Optional[int] = None,
    extrapolate: bool = True,
) -> dict[int, Optional[np.ndarray]]:
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
    max_gap : int, optional
        Maximum number of frames to interpolate across. If a gap between detections
        exceeds this, it is filled with None instead of interpolating. Defaults to None.
    extrapolate : bool, optional
        Whether to extrapolate at the beginning (before first detection) and end
        (after last detection). If False, these periods are filled with None.
        Default behavior is to extrapolate at the beginning and end.
        Defaults to True.

    Returns
    -------
    dict[int, Optional[numpy.ndarray]]
        A dictionary with a 3x3 homography for every frame from 0..max_frame.
        Keys will be standard Python int. Values can be 3x3 np.ndarray or None
        for frames in gaps exceeding max_gap or before/after detections (if extrapolate=False).
    """
    # convert np.int64 keys to int for sorting
    homographies_fixed_keys = {}
    for frame_idx, H in homographies_dict.items():
        homographies_fixed_keys[int(frame_idx)] = H

    # sort frames that have known homographies
    known_frames = sorted(homographies_fixed_keys.keys())
    if not known_frames:
        # No known frames => return empty dict
        return {}

    upsampled = {}

    # fill frames before the first known up to frame 0 (extrapolate)
    first_frame = known_frames[0]
    if extrapolate:
        for f in range(0, first_frame):
            upsampled[f] = homographies_fixed_keys[first_frame]
    else:
        for f in range(0, first_frame):
            upsampled[f] = None

    # interpolate between consecutive known frames
    for i in tqdm(range(len(known_frames) - 1), desc="Interpolating homographies"):
        f1 = known_frames[i]
        f2 = known_frames[i + 1]

        H1 = homographies_fixed_keys[f1]  # 3x3
        H2 = homographies_fixed_keys[f2]  # 3x3

        frame_diff = f2 - f1

        # Check if gap exceeds max_gap threshold
        if max_gap is not None and frame_diff > max_gap:
            # Fill gap with None instead of interpolating
            for offset in range(frame_diff):
                current_frame = f1 + offset
                if current_frame > max_frame:
                    break
                if offset == 0:
                    upsampled[current_frame] = H1
                else:
                    upsampled[current_frame] = None
        else:
            # Fill from f1..(f2-1) with interpolation
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

    # Fill frames beyond the last known up to max_frame (extrapolate)
    if extrapolate:
        for f in range(last_frame + 1, max_frame + 1):
            upsampled[f] = homographies_fixed_keys[last_frame]
    else:
        for f in range(last_frame + 1, max_frame + 1):
            upsampled[f] = None

    return upsampled


def _resample_homographies_to_gaze(
    homographies_df: pd.DataFrame, gaze_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Resample a DataFrame of homographies to the timestamps of a gaze DataFrame.

    Parameters
    ----------
    homographies_df : pd.DataFrame
        DataFrame with 'timestamp [ns]' as index and 'homography' column (3x3 np.ndarray).
    gaze_df : pd.DataFrame
        Gaze DataFrame whose index will be used for resampling.

    Returns
    -------
    pd.DataFrame
        Resampled homographies DataFrame indexed by gaze timestamps.
    """
    # Create a DataFrame of homography elements for interpolation
    h_elements = []
    for ts, row in homographies_df.iterrows():
        H = row["homography"]
        if H is not None:
            record = {"timestamp [ns]": ts, "frame_idx": row["frame_idx"]}
            for r in range(3):
                for c in range(3):
                    record[f"h{r}{c}"] = H[r, c]
            h_elements.append(record)

    if not h_elements:
        # If no homographies found, return DataFrame with gaze index and NaNs
        resampled_h = pd.DataFrame(index=gaze_df.index)
        resampled_h.index.name = "timestamp [ns]"
        resampled_h["frame_idx"] = np.nan
        resampled_h["homography"] = None
        return resampled_h

    h_df = pd.DataFrame(h_elements).set_index("timestamp [ns]")

    from ..preprocess import interpolate

    resampled_h = interpolate(gaze_df.index.to_numpy(), h_df)

    # Reconstruct homography matrices
    hs = [resampled_h[f"h{r}{c}"].values for r in range(3) for c in range(3)]
    h_stack = np.stack(hs, axis=1)  # (N, 9)

    # Handle NaNs: one None matrix for any row containing NaN
    any_nan = np.any(np.isnan(h_stack), axis=1)
    h_matrices = [None] * len(resampled_h)
    for i in range(len(resampled_h)):
        if not any_nan[i]:
            h_matrices[i] = h_stack[i].reshape(3, 3)

    resampled_h["homography"] = h_matrices
    resampled_h = resampled_h[["frame_idx", "homography"]]
    return resampled_h
