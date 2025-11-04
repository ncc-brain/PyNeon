from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from ..stream import Stream

if TYPE_CHECKING:
    from ..recording import Recording


def overlay_detections_and_pose(
    recording: "Recording",
    april_detections: pd.DataFrame,
    camera_positions: pd.DataFrame,
    room_corners: np.ndarray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
    video_output_path: Path | str = "detection_and_pose.mp4",
    graph_size: np.ndarray = np.array([300, 300]),
    show_video: bool = True,
):
    """
    Overlay AprilTag detections and camera positions on top of the recording's video frames.
    Additionally, draw an inset plot showing the camera's trajectory within known environmental
    (room) boundaries.

    This function reads frames from the provided recording video and overlays:
    - AprilTag detections (if present in the current frame)
    - The camera position, using a mini-map inset showing all previously visited positions,
    as well as the current/last known position of the camera.

    The mini-map inset uses predetermined room boundaries derived from the provided
    `room_corners` array to map positions onto a fixed coordinate system, allowing consistent
    visualization of the camera's movement over time.

    Parameters
    ----------

    recording :
        Recording object containing the video and related metadata.
    april_detections : pandas.DataFrame
        DataFrame containing AprilTag detections for each frame, with columns:
            - 'frame_idx': int
                The frame number.
            - 'tag_id': int
                The ID of the detected AprilTag.
            - 'corners': np.ndarray of shape (4,2)
                Pixel coordinates of the tag's corners.
    camera_positions : :pandas.DataFrame
        DataFrame containing the camera positions for each frame, with at least:
            - 'frame_idx': int
                The frame number.
            - 'smoothed_camera_pos': numpy.ndarray of shape (3,)
                The camera position [x, y, z] in world coordinates.
    room_corners : numpy.ndarray of shape (N, 2), optional
        Array defining the polygon corners of the room in world coordinates.
        Defaults to a simple unit square: [[0,0],[0,1],[1,1],[1,0]].
    video_output_path : str or pathlib.Path, optional
        Path to save the output video with overlays. Defaults to 'output_with_overlays.mp4'.
    graph_size : numpy.ndarray of shape (2,), optional
        The width and height (in pixels) of the inset mini-map. Defaults to [300, 300].
    show_video : bool, optional
        Whether to display the video with overlays as it is processed. Press 'ESC' to stop early.
        Defaults to True.

    Notes
    -----

    - If the video cannot be read, a RuntimeError is raised.
    - Press 'ESC' to stop playback if show_video is True.

    """

    # Compute the room boundaries from the provided corners
    room_min_x = np.min(room_corners[:, 0])
    room_max_x = np.max(room_corners[:, 0])
    room_min_y = np.min(room_corners[:, 1])
    room_max_y = np.max(room_corners[:, 1])

    # Extract camera positions into a dictionary for quick lookup
    results_dict = {
        row["frame_idx"]: row["camera_pos"] for _, row in camera_positions.iterrows()
    }

    # Group detections by frame
    detections_by_frame = {}
    for _, row in april_detections.iterrows():
        fidx = row["frame_idx"]
        if fidx not in detections_by_frame:
            detections_by_frame[fidx] = []
        detections_by_frame[fidx].append((row["tag_id"], row["corners"]))

    cap = recording.video
    cap.reset()
    frame_idx = 0

    # Track last known detections and positions
    last_detections = None
    last_position = None
    ever_detected = False

    visited_positions = []

    # Extract graph dimensions
    graph_width, graph_height = graph_size

    def draw_detections(frame, detections, color, thickness=2):
        for tag_id, corners in detections:
            # Ensure corners are a NumPy array before converting to int
            corners_array = np.array(corners, dtype=np.float32)
            corners_int = corners_array.astype(int)

            cv2.polylines(frame, [corners_int], True, color, thickness)
            for c in corners_int:
                cv2.circle(frame, tuple(c), 4, color, -1)
            corner_text_pos = (corners_int[0, 0], corners_int[0, 1] - 10)
            cv2.putText(
                frame,
                f"ID: {tag_id}",
                corner_text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    def position_to_graph_coords(position, x0, y0, w, h, min_x, max_x, min_y, max_y):
        x, y, z = position
        x_norm = (x - min_x) / (max_x - min_x)
        y_norm = (y - min_y) / (max_y - min_y)
        pt_x = int(x0 + x_norm * w)
        pt_y = int(y0 + (1 - y_norm) * h)
        return (pt_x, pt_y)

    def draw_coordinate_cross(frame, x0, y0, w, h, min_x, max_x, min_y, max_y):
        # Draw a black background
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)

        # Only draw axes if 0,0 is within the range
        if min_x < 0 < max_x and min_y < 0 < max_y:
            origin = position_to_graph_coords(
                (0, 0, 0), x0, y0, w, h, min_x, max_x, min_y, max_y
            )

            line_color = (200, 200, 200)
            thickness = 1

            # Draw x-axis line
            cv2.line(frame, (x0, origin[1]), (x0 + w, origin[1]), line_color, thickness)

            # Draw y-axis line
            cv2.line(frame, (origin[0], y0), (origin[0], y0 + h), line_color, thickness)

    def draw_mini_graph(
        frame, current_position, detected, visited_positions, min_x, max_x, min_y, max_y
    ):
        h, w = frame.shape[:2]
        x0 = w - graph_width - 10
        y0 = h - graph_height - 10

        # Draw axes and background
        draw_coordinate_cross(
            frame, x0, y0, graph_width, graph_height, min_x, max_x, min_y, max_y
        )

        # Draw visited positions in dim color
        dim_color = (100, 100, 100)
        for pos in visited_positions:
            pt = position_to_graph_coords(
                pos, x0, y0, graph_width, graph_height, min_x, max_x, min_y, max_y
            )
            cv2.circle(frame, pt, 3, dim_color, -1)

        # Draw current/last position
        if current_position is not None:
            pt = position_to_graph_coords(
                current_position,
                x0,
                y0,
                graph_width,
                graph_height,
                min_x,
                max_x,
                min_y,
                max_y,
            )
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.circle(frame, pt, 5, color, -1)

    # Try reading a frame to determine size
    ret, test_frame = cap.read()
    height, width = cap.height, cap.width
    fps = cap.fps or 30

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_output_path = Path(video_output_path)
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        current_detections = detections_by_frame.get(frame_idx, None)
        current_position = results_dict.get(frame_idx, None)

        if current_position is not None:
            visited_positions.append(current_position)

        if current_detections is not None:
            # Current frame has detections
            ever_detected = True
            last_detections = current_detections
            last_position = (
                current_position if current_position is not None else last_position
            )

            draw_detections(frame, current_detections, (0, 255, 0))
            draw_mini_graph(
                frame,
                current_position,
                True,
                visited_positions,
                room_min_x,
                room_max_x,
                room_min_y,
                room_max_y,
            )
        else:
            # No current detections
            if ever_detected and last_detections is not None:
                # Draw last known detections in red
                draw_detections(frame, last_detections, (0, 0, 255))
                draw_mini_graph(
                    frame,
                    last_position,
                    False,
                    visited_positions,
                    room_min_x,
                    room_max_x,
                    room_min_y,
                    room_max_y,
                )
            else:
                # Never had detections: draw a green overlay
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (frame.shape[1], frame.shape[0]),
                    (0, 255, 0),
                    thickness=20,
                )
                alpha = 0.2
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                draw_mini_graph(
                    frame,
                    last_position,
                    True,
                    visited_positions,
                    room_min_x,
                    room_max_x,
                    room_min_y,
                    room_max_y,
                )

        if show_video:
            # Resize the frame for display
            resized_frame = cv2.resize(
                frame, (width // 2, height // 2)
            )  # Adjust scaling factor as needed
            cv2.imshow("Video with Overlays", resized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        out.write(frame)
        frame_idx += 1

    out.release()
    cv2.destroyAllWindows()
