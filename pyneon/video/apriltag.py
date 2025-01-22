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

    # convert to pandas DataFrame
    all_detections = pd.DataFrame(all_detections)

    return all_detections


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
