from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.doc_decorators import fill_doc
from ..video.constants import DETECTION_COLUMNS

if TYPE_CHECKING:
    from ..recording import Recording
    from ..stream import Stream
    from ..video import Video


@fill_doc
def plot_frame(
    video: "Video",
    frame_index: int = 0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a frame from the video on a matplotlib axis.

    Parameters
    ----------
    video : Video
        Video instance to plot the frame from.
    frame_index : int
        Index of the frame to plot.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_return)s
    """
    if frame_index >= len(video.ts) or frame_index < 0:
        raise IndexError(f"Frame index {frame_index} out of range")
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    frame = video.read_frame_at(frame_index)
    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    if show:
        plt.show()

    return fig, ax


def _plot_marker_detections(
    ax: plt.Axes,
    detections: pd.DataFrame,
    show_ids: bool,
    color: str,
) -> None:
    for _, marker in detections.iterrows():
        if show_ids:
            ax.text(
                marker["center x [px]"],
                marker["center y [px]"],
                marker["marker id"],
                color=color,
                ha="center",
                va="center",
            )
        corners_x = [
            marker["top left x [px]"],
            marker["top right x [px]"],
            marker["bottom right x [px]"],
            marker["bottom left x [px]"],
            marker["top left x [px]"],
        ]
        corners_y = [
            marker["top left y [px]"],
            marker["top right y [px]"],
            marker["bottom right y [px]"],
            marker["bottom left y [px]"],
            marker["top left y [px]"],
        ]
        ax.plot(corners_x, corners_y, color=color)


def _plot_corner_detections(
    ax: plt.Axes,
    detections: pd.DataFrame,
    show_ids: bool,
    color: str,
) -> None:
    for _, detection in detections.iterrows():
        corners = np.asarray(detection["corners"], dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(
                f"Detected corners must have shape (4, 2), got {corners.shape}"
            )
        corners_x = list(corners[:, 0]) + [corners[0, 0]]
        corners_y = list(corners[:, 1]) + [corners[0, 1]]
        ax.plot(corners_x, corners_y, color=color)

        if show_ids and "marker id" in detection:
            if "center" in detection and detection["center"] is not None:
                center = np.asarray(detection["center"], dtype=np.float32).reshape(-1)
            else:
                center = np.mean(corners, axis=0)
            ax.text(
                center[0],
                center[1],
                int(detection["marker id"]),
                color=color,
                ha="center",
                va="center",
            )


@fill_doc
def plot_detections(
    video: "Video",
    detections: "Stream",
    frame_index: int,
    show_ids: bool = True,
    color: str = "magenta",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot detections on a frame from the video.

    Parameters
    ----------
    video : Video
        Video instance to plot the frame from.
    detections : Stream
        Stream containing marker or surface-corner detections.
    frame_index : int
        Index of the frame to plot.
    show_ids : bool
        Display detection IDs at their centers when available.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
        %(fig_ax_return)s
    """
    fig, ax = plot_frame(video, frame_index=frame_index, ax=ax, show=False)
    detections_df = detections.data
    if "frame index" not in detections_df.columns:
        raise ValueError("Detections must include a 'frame index' column.")
    frame_detections = detections_df[detections_df["frame index"] == frame_index]

    if not frame_detections.empty:
        if set(DETECTION_COLUMNS).issubset(frame_detections.columns):
            _plot_marker_detections(ax, frame_detections, show_ids, color)
        elif "corners" in frame_detections.columns:
            _plot_corner_detections(ax, frame_detections, show_ids, color)
        else:
            raise ValueError(
                "Detections must contain marker corner columns or a 'corners' column."
            )
    if show:
        plt.show()
    return fig, ax


@fill_doc
def overlay_detections(
    video: "Video",
    detections: "Stream",
    show_ids: bool = True,
    color: tuple[int, int, int] = (255, 0, 255),
    show_video: bool = False,
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Overlay detections on the video frames and display or save the video with overlays.

    Parameters
    ----------
    video : Video
        Video instance to overlay the detected markers on.
    detections : Stream
        Stream containing marker or surface-corner detections.
    show_ids : bool
        Whether to overlay IDs at their centers when available. Defaults to True.
    color : tuple[int, int, int]
        BGR color tuple for marker overlays. Defaults to (255, 0, 255) which is magenta.
    %(show_video_param)s
    %(output_path_param)s
        Defaults to 'detected_markers.mp4'.
    """
    # Either show video or save it
    if output_path is None and not show_video:
        raise ValueError(
            "Either show_video=True or output_path must be provided."
        )

    # Reset video to the beginning
    video.reset()

    # Initialize video writer if saving
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(output_path), fourcc, video.fps, (video.width, video.height)
        )

    detections_df = detections.data
    if "frame index" not in detections_df.columns:
        raise ValueError("Detections must include a 'frame index' column.")
    is_marker = set(DETECTION_COLUMNS).issubset(detections_df.columns)
    uses_corners = "corners" in detections_df.columns
    if not is_marker and not uses_corners:
        raise ValueError(
            "Detections must contain marker corner columns or a 'corners' column."
        )

    grouped = detections_df.groupby("frame index")
    detections_by_frame = {frame_idx: group for frame_idx, group in grouped}

    # Iterate through video frames sequentially
    for frame_index in tqdm(
        range(len(video.ts)),
        desc="Overlaying detections on video",
        total=len(video.ts),
    ):
        # Read the next frame sequentially
        ret, frame = video.read()
        if not ret:
            print(f"frame {frame_index} is skipped")
            break

        # Get markers for this frame (only display if markers are actually detected)
        if frame_index not in detections_by_frame:
            # No markers detected in this frame, skip overlay
            if show_video:
                cv2.namedWindow("Detections Overlay", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    "Detections Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
                cv2.imshow("Detections Overlay", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if output_path is not None:
                out.write(frame)
            continue

        frame_detections = detections_by_frame[frame_index]

        # Draw each detection
        for _, detection in frame_detections.iterrows():
            if is_marker:
                corners = np.array(
                    [
                        [detection["top left x [px]"], detection["top left y [px]"]],
                        [
                            detection["top right x [px]"],
                            detection["top right y [px]"],
                        ],
                        [
                            detection["bottom right x [px]"],
                            detection["bottom right y [px]"],
                        ],
                        [
                            detection["bottom left x [px]"],
                            detection["bottom left y [px]"],
                        ],
                    ],
                    dtype=np.int32,
                )
                center = np.array(
                    [detection["center x [px]"], detection["center y [px]"]],
                    dtype=np.int32,
                ).reshape(-1)
                label = str(int(detection["marker id"])) if show_ids else None
            else:
                corners = np.asarray(detection["corners"], dtype=np.float32)
                if corners.shape != (4, 2):
                    raise ValueError(
                        f"Detected corners must have shape (4, 2), got {corners.shape}"
                    )
                corners = corners.astype(np.int32)
                if "center" in detection and detection["center"] is not None:
                    center = np.asarray(detection["center"], dtype=np.float32).reshape(
                        -1
                    )
                else:
                    center = np.mean(corners, axis=0)
                label = (
                    str(int(detection["marker id"]))
                    if show_ids and "marker id" in detection
                    else None
                )

            cv2.polylines(frame, [corners], True, color, 2)

            if label is not None:
                text = label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                text_x = int(center[0]) - text_width // 2
                text_y = int(center[1]) + text_height // 2

                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness,
                )

        # Display the frame if requested
        if show_video:
            cv2.namedWindow("Detections Overlay", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "Detections Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow("Detections Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write the frame to output video if saving
        if output_path is not None:
            out.write(frame)

    # Release resources
    if output_path is not None:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    video.reset()


def overlay_scanpath(
    video: "Video",
    scanpath: pd.DataFrame,
    circle_radius: int = 10,
    line_thickness: int = 2,
    text_size: Optional[int] = None,
    max_fixations: int = 10,
    show_video: bool = False,
    output_path: Optional[Path | str] = "scanpath.mp4",
) -> None:
    """
    Plot scanpath on top of the video frames. The resulting video can be displayed and/or saved.

    Parameters
    ----------
    video : Video
        Video instance to overlay the fixations on.
    scanpath : pandas.DataFrame
        DataFrame containing the fixations and gaze data.
    circle_radius : int
        Radius of the fixation circles in pixels. Defaults to 10.
    line_thickness : int or None
        Thickness of the lines connecting fixations. If None, no lines are drawn.
        Defaults to 2.
    text_size : int or None
        Size of the text displaying fixation status and ID. If None, no text is displayed.
        Defaults to None.
    max_fixations : int
        Maximum number of fixations to plot per frame. Defaults to 10.
    show_video : bool
        Whether to display the video with fixations overlaid. Defaults to False.
    output_path : pathlib.Path or str or None
        Path to save the video with fixations overlaid. If None, the video is not saved.
        Defaults to 'scanpath.mp4'.
    """
    # Either show video or save it
    if output_path is None and not show_video:
        raise ValueError(
            "Either show_video=True or output_path must be provided."
        )

    # Check scanpath DataFrame
    if "fixations" not in scanpath.columns:
        raise ValueError("scanpath DataFrame must contain a 'fixations' column.")
    if not np.allclose(scanpath.index, video.ts):
        raise ValueError("Gaze and video timestamps do not match.")

    # reset video to the beginning
    video.reset()

    # Initialize video capture and writer
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            output_path, fourcc, video.fps, (video.width, video.height)
        )

    # Iterate through each row in the tracked fixations DataFrame
    for idx, row in tqdm(
        scanpath.iterrows(),
        desc="Plotting scanpath on scene video",
        total=scanpath.shape[0],
    ):
        # Read the current frame from the video
        ret, frame = video.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from the video.")

        # Extract fixations and gaze data
        fixations = row["fixations"]
        prev_x, prev_y = None, None

        # check if fixations is empty
        if fixations.empty:
            continue

        for i in fixations.index:
            x, y, status, id = fixations.loc[
                i, ["gaze x [px]", "gaze y [px]", "fixation status", "fixation id"]
            ].values

            # pass if status or id are nan
            if pd.isna(status) or pd.isna(id) or i > max_fixations:
                continue

            else:
                # Set color based on fixation status
                if status == "tracked":
                    color = (0, 255, 0)  # Green for tracked
                elif status == "lost":
                    color = (0, 0, 255)  # Red for lost
                else:
                    color = (255, 0, 0)  # Blue for other statuses

                # Draw fixation circles and connecting lines
                if pd.notna(x) and pd.notna(y):
                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        radius=circle_radius,
                        color=color,
                        thickness=-1,
                    )

                    # Optionally add text showing fixation status and ID
                    if text_size is not None:
                        cv2.putText(
                            frame,
                            f"ID: {id} Status: {status}",
                            (int(x) + 10, int(y)),
                            cv2.FONT_HERSHEY_PLAIN,
                            text_size,
                            color,
                            text_size,
                        )

                    # Draw line connecting the previous fixation to the current one
                    if (
                        pd.notna(prev_x)
                        and pd.notna(prev_y)
                        and line_thickness is not None
                    ):
                        cv2.line(
                            frame,
                            (int(prev_x), int(prev_y)),
                            (int(x), int(y)),
                            color,
                            line_thickness,
                        )

                    # Update the previous fixation point
                    prev_x, prev_y = x, y

        # Display the frame with overlays (Optional)
        if show_video:
            cv2.namedWindow("Fixations Overlay", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "Fixations Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow("Fixations Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write the frame with overlays to the output video
        if output_path is not None:
            out.write(frame)

    # Release resources
    out.release()
    cv2.destroyAllWindows()
    video.reset()


def overlay_detections_and_pose(
    recording: "Recording",
    april_detections: pd.DataFrame,
    camera_positions: pd.DataFrame,
    room_corners: np.ndarray = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
    output_path: Path | str = "detection_and_pose.mp4",
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
        Recording instance containing the video and related metadata.
    april_detections : pandas.DataFrame
        DataFrame containing AprilTag detections for each frame, with columns:
            - 'frame id': int
                The frame number.
            - "marker id": int
                The ID of the detected AprilTag.
            - "corners": np.ndarray of shape (4,2)
                Pixel coordinates of the marker's corners.
    camera_positions : :pandas.DataFrame
        DataFrame containing the camera positions for each frame, with at least:
            - 'frame id': int
                The frame number.
            - 'smoothed_camera_pos': numpy.ndarray of shape (3,)
                The camera position [x, y, z] in world coordinates.
    room_corners : numpy.ndarray of shape (N, 2), optional
        Array defining the polygon corners of the room in world coordinates.
        Defaults to a simple unit square: [[0,0],[0,1],[1,1],[1,0]].
    output_path : str or pathlib.Path, optional
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
        row["frame index"]: row["camera_pos"] for _, row in camera_positions.iterrows()
    }

    # Group detections by frame
    detections_by_frame = {}
    for _, row in april_detections.iterrows():
        f_id = row["frame index"]
        if f_id not in detections_by_frame:
            detections_by_frame[f_id] = []
        # Reconstruct corners array from individual columns
        corners = np.array(
            [
                [row["corner 0 x [px]"], row["corner 0 y [px]"]],
                [row["corner 1 x [px]"], row["corner 1 y [px]"]],
                [row["corner 2 x [px]"], row["corner 2 y [px]"]],
                [row["corner 3 x [px]"], row["corner 3 y [px]"]],
            ]
        )
        detections_by_frame[f_id].append((row["marker id"], corners))

    cap = recording.scene_video
    cap.reset()
    frame_index = 0

    # Track last known detections and positions
    last_detections = None
    last_position = None
    ever_detected = False

    visited_positions = []

    # Extract graph dimensions
    graph_width, graph_height = graph_size

    def draw_detections(frame, detections, color, thickness=2):
        for marker_id, corners in detections:
            # Ensure corners are a NumPy array before converting to int
            corners_array = np.array(corners, dtype=np.float32)
            corners_int = corners_array.astype(int)

            cv2.polylines(frame, [corners_int], True, color, thickness)
            for c in corners_int:
                cv2.circle(frame, tuple(c), 4, color, -1)
            corner_text_pos = (corners_int[0, 0], corners_int[0, 1] - 10)
            cv2.putText(
                frame,
                f"ID: {marker_id}",
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
    output_path = Path(output_path)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        current_detections = detections_by_frame.get(frame_index, None)
        current_position = results_dict.get(frame_index, None)

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
            cv2.namedWindow("Video with Overlays", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "Video with Overlays", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow("Video with Overlays", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
