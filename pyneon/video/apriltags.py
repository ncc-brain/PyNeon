import cv2
import numpy as np
import pandas as pd
import os
import pickle
from typing import TYPE_CHECKING, Union, Optional, Dict, List
from pupil_apriltags import Detector


if TYPE_CHECKING:
    from ..recording import NeonRecording
    from .video import NeonVideo


def detect_apriltags(video: "NeonVideo", tag_family: str = "tag36h11"):
    """
    Detect AprilTags in a video and report their data for every frame using the apriltag library.

    Parameters
    ----------
    video : cv2.VideoCapture or similar video object
        A video capture object from which frames can be read.
    tag_family : str, optional
        The AprilTag family to detect (default is 'tag36h11').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing AprilTag detections, with columns:
        - 'frame_idx': The frame number
        - 'tag_id': The ID of the detected AprilTag
        - 'corners': A 4x2 array of the tag corner coordinates
        - 'center': A 1x2 array with the tag center coordinates
    """

    # Initialize the detector with the specified tag family
    detector = Detector(families=tag_family)

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not video.isOpened():
        raise ValueError("Unable to open video")

    all_detections = []
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to grayscale for detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = detector.detect(gray_frame)

        for detection in detections:
            # Extract the tag ID and corners
            tag_id = detection.tag_id
            corners = detection.corners

            # Calculate the center of the tag
            center = np.mean(corners, axis=0)

            # Store the detection data
            all_detections.append(
                {
                    "frame_idx": frame_idx,
                    "tag_id": tag_id,
                    "corners": corners,
                    "center": center,
                }
            )

        frame_idx += 1

    video.release()

    # convert to pandas DataFrame
    all_detections = pd.DataFrame(all_detections)

    return all_detections


def compute_camera_positions(
    video: "NeonVideo",
    tag_locations: Dict[int, List[float]],
    tag_size: Union[float, Dict[int, float]],
    all_detections: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Compute the camera position for each frame using AprilTag detections stored in a DataFrame.

    This function uses a pinhole camera model and the `cv2.solvePnP` function to determine the
    camera translation and rotation vectors for each frame. The known 3D positions of each tag
    and the detected 2D corners are used as correspondences. The resulting camera pose is then
    expressed in world coordinates.

    Parameters
    ----------
    all_detections : pd.DataFrame
        DataFrame containing AprilTag detections. Expected columns:
        - 'frame_idx': int, frame number
        - 'tag_id': int, ID of the detected tag
        - 'corners': np.ndarray of shape (4, 2), pixel coordinates of the tag's corners.
        - 'center': np.ndarray of shape (1, 2), pixel coordinates of the tag's center (not strictly required here).
    tag_locations : dict
        Dictionary mapping tag IDs to their known 3D center coordinates in the world frame.
        Example: { tag_id: [X, Y, Z] }
    tag_size : float or dict
        If a float is provided, all tags are assumed to have the same size (side length in meters).
        If a dict is provided, it should map tag IDs to their respective side lengths.
        Example: 0.075 (for a single size) or {0: 0.075, 1: 0.1} for different sizes.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per frame containing:
        - 'frame_idx': The frame number.
        - 'translation_vector': (3,) array representing camera translation in world coordinates.
        - 'rotation_vector': (3,) array representing camera rotation in Rodrigues form.
        - 'camera_pos': (3,) array representing the camera position in world coordinates.

    Notes
    -----
    - The camera's intrinsic parameters are estimated from the field of view and resolution.
        For more accurate results, use known camera intrinsics.
    - The function assumes that each tag's known location is provided as the center of the tag,
        and constructs the tag's corners in 3D world coordinates by offsetting from its center.

    Raises
    ------
    ValueError
        If no sufficient points are found to solve PnP for a given frame.
    """

    # check if all_detections is empty
    if all_detections.empty:
        # call the detect_apriltags function to get the detections
        all_detections = detect_apriltags(video)
        # if still empty, return an empty DataFrame
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

    # Handle tag size inputs
    if isinstance(tag_size, float):
        # Single size for all tags
        def get_tag_half_size(tid):
            return tag_size / 2.0
    elif isinstance(tag_size, dict):
        # Different sizes per tag
        def get_tag_half_size(tid):
            if tid not in tag_size:
                raise ValueError(
                    f"Tag ID {tid} not found in provided tag_size dictionary."
                )
            return tag_size[tid] / 2.0
    else:
        raise TypeError(
            "tag_size must be either a float or a dictionary mapping tag IDs to sizes."
        )

    camera_matrix = video.camera_matrix
    dist_coeffs = video.dist_coeffs

    # If no distortion is provided, assume zero distortion
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    results = []

    # Process each unique frame
    for frame in all_detections["frame_idx"].unique():
        frame_detections = all_detections.loc[all_detections["frame_idx"] == frame]

        if frame_detections.empty:
            # No tags detected in this frame, skip
            continue

        object_points = []
        image_points = []

        # Collect all object-image correspondences for this frame
        for _, row in frame_detections.iterrows():
            tag_id = row["tag_id"]
            corners = row["corners"]  # shape (4,2)

            if tag_id not in tag_locations:
                # If no known location for this tag is provided, skip it
                continue

            tag_center_3d = np.array(tag_locations[tag_id], dtype=np.float32)
            half_size = get_tag_half_size(tag_id)

            # Compute the 3D corners of the tag from its center
            # The tag plane orientation is assumed. Adjust as needed.
            tag_3d_corners = np.array(
                [
                    [
                        tag_center_3d[0],
                        tag_center_3d[1] - half_size,
                        tag_center_3d[2] + half_size,
                    ],
                    [
                        tag_center_3d[0],
                        tag_center_3d[1] + half_size,
                        tag_center_3d[2] + half_size,
                    ],
                    [
                        tag_center_3d[0],
                        tag_center_3d[1] + half_size,
                        tag_center_3d[2] - half_size,
                    ],
                    [
                        tag_center_3d[0],
                        tag_center_3d[1] - half_size,
                        tag_center_3d[2] - half_size,
                    ],
                ],
                dtype=np.float32,
            )

            object_points.extend(tag_3d_corners)
            image_points.extend(corners)

        if len(object_points) < 4:
            # Not enough points to solve for pose
            continue

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Solve the PnP problem to find rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )

        if not success:
            # Could not solve for this frame
            continue

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rotation_vector)

        # Compute camera position in world coordinates
        # World to camera: Pc = R * Pw + t
        # Pw = R^T * (Pc - t), with Pc=0 (camera center)
        # camera_pos = -R^T * t
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
