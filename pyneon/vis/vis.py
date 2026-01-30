from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ..utils.doc_decorators import fill_doc
from ..video.utils import marker_family_to_dict

if TYPE_CHECKING:
    from ..epochs import Epochs
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
    video : SceneVideo
        Video object to plot the frame from.
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
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if ret:
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    else:
        raise RuntimeError(f"Could not read frame {frame_index}")
    if show:
        plt.show()

    video.reset()
    return fig, ax


@fill_doc
def plot_detected_markers(
    video: "Video",
    detected_markers: "Stream",
    frame_index: int = 0,
    show_marker_ids: bool = True,
    color: str = "magenta",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot detected markers on a frame from the video.

    Parameters
    ----------
    video : SceneVideo
        Video object to plot the frame from.
    detected_markers : Stream
        Stream containing detected marker data.
        See :meth:`pyneon.video.detect_markers` for details.
    frame_index : int
        Index of the frame to plot.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
        %(fig_ax_return)s
    """
    fig, ax = plot_frame(video, frame_index=frame_index, ax=ax, show=False)

    mask = detected_markers["frame index"] == frame_index
    markers = detected_markers.data[mask]

    for _, marker in markers.iterrows():
        if show_marker_ids:
            ax.text(
                marker["center x [px]"],
                marker["center y [px]"],
                marker["marker id"],
                color=color,
                ha="center",
                va="center",
            )
        # Draw marker corners with connected lines
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
    if show:
        plt.show()
    return fig, ax


@fill_doc
def plot_marker_layout(
    marker_layout: pd.DataFrame,
    show_marker_names: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D marker layout on a matplotlib axis.

    This renders the marker images on a white canvas using the marker metadata
    in ``marker_layout`` and overlays marker names at their centers.

    Parameters
    ----------
    %(marker_layout)s
    show_marker_names : bool
        Whether to overlay marker names at their centers. Defaults to True.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_return)s
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Calculate actual bounds for each marker, then find overall canvas size
    min_x = (marker_layout["center x"] - marker_layout["size"] / 2).min()
    max_x = (marker_layout["center x"] + marker_layout["size"] / 2).max()
    min_y = (marker_layout["center y"] - marker_layout["size"] / 2).min()
    max_y = (marker_layout["center y"] + marker_layout["size"] / 2).max()

    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)

    # Create white background
    canvas = Image.new("L", (canvas_width, canvas_height), color=255)

    for _, marker in marker_layout.iterrows():
        marker_name = marker["marker name"]
        marker_family = marker_name.rsplit("_", 1)[0]
        marker_id = int(marker_name.rsplit("_", 1)[1])

        marker_type, aruco_dict = marker_family_to_dict(marker_family)
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict, marker_id, int(marker["size"])
        )
        if marker_type == "april":
            # flip image diagonally for AprilTag markers
            # because AprilTag has a different orientation than ArUco
            # see https://github.com/opencv/opencv-python/issues/1195
            marker_img = np.flipud(np.fliplr(marker_img))

        # Convert to PIL Image
        marker_pil = Image.fromarray(marker_img)

        # Calculate paste position (top-left corner)
        x = int(marker["center x"] - marker["size"] / 2 - min_x)
        y = int(marker["center y"] - marker["size"] / 2 - min_y)

        # Paste marker onto canvas
        canvas.paste(marker_pil, (x, y))
        
        if show_marker_names:
            ax.text(
                marker["center x"],
                marker["center y"],
                marker_name,
                color="red",
                ha="center",
                va="center",
            )

    # Display the canvas
    ax.imshow(canvas, cmap="gray", extent=[min_x, max_x, max_y, min_y])
    ax.set_xlabel("x [surface coord]")
    ax.set_ylabel("y [surface coord]")
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax


@fill_doc
def plot_distribution(
    rec: "Recording",
    heatmap_source: Literal["gaze", "fixations", None] = "gaze",
    scatter_source: Literal["gaze", "fixations", None] = "fixations",
    step_size: int = 10,
    sigma: int | float = 2,
    width_height: tuple[int, int] = (1600, 1200),
    cmap: str = "inferno",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of gaze or fixation data on a matplotlib axis.
    Users can flexibly choose to generate a smoothed heatmap and/or scatter plot and
    the source of the data (gaze or fixation).

    Parameters
    ----------
    rec : Recording
        Recording object containing the gaze and video data.
    heatmap_source : {'gaze', 'fixations', None}
        Source of the data to plot as a heatmap. If None, no heatmap is plotted.
        Defaults to 'gaze'.
    scatter_source : {'gaze', 'fixations', None}
        Source of the data to plot as a scatter plot. If None, no scatter plot is plotted.
        Defaults to 'fixations'. Gaze data is typically more dense and thus less suitable
        for scatter plots.
    step_size : int
        Size of the grid cells in pixels. Defaults to 10.
    sigma : int or float
        Standard deviation of the Gaussian kernel used to smooth the heatmap.
        If None or 0, no smoothing is applied. Defaults to 2.
    width_height : tuple[int, int]
        If video is not available, the width and height of the scene camera frames to
        specify the heatmap dimensions. Defaults to (1600, 1200).
    cmap : str
        Colormap to use for the heatmap. Defaults to 'inferno'.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_return)s
    """
    if heatmap_source is None and scatter_source is None:
        raise ValueError(
            "At least one of heatmap_source and scatter_source must be provided."
        )

    if rec.scene_video is None:
        if width_height is None:
            raise ValueError("No video data available and no width_height provided.")
        width_height = width_height
    else:
        width_height = (rec.scene_video.width, rec.scene_video.height)

    if rec.gaze is None:
        raise ValueError("No gaze data available.")
    if rec.fixations is None:
        raise ValueError("No fixation data available.")
    gaze = rec.gaze
    fixations = rec.fixations

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Generate the heatmap data
    x_edges = np.linspace(0, width_height[0], width_height[0] // step_size + 1)
    y_edges = np.linspace(0, width_height[1], width_height[1] // step_size + 1)

    # Sample gaze and fixation data (replace with your actual data)
    gaze_x = gaze.data["gaze x [px]"]
    gaze_y = gaze.data["gaze y [px]"]
    fix_x = fixations.data["fixation x [px]"]
    fix_y = fixations.data["fixation y [px]"]

    heatmap_x = gaze_x if heatmap_source == "gaze" else fix_x
    heatmap_y = gaze_y if heatmap_source == "gaze" else fix_y
    scatter_x = gaze_x if scatter_source == "gaze" else fix_x
    scatter_y = gaze_y if scatter_source == "gaze" else fix_y

    heatmap, _, _ = np.histogram2d(heatmap_x, heatmap_y, bins=(x_edges, y_edges))
    if sigma is not None or sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)

    if heatmap_source is not None:
        h = ax.imshow(
            heatmap.T,
            cmap=cmap,
            extent=[0, width_height[0], 0, width_height[1]],
            origin="lower",
        )
        plt.colorbar(h, ax=ax, label="Density")
    if scatter_source is not None:
        ax.scatter(scatter_x, scatter_y, color="white", s=10, alpha=0.3, edgecolors="k")
    ax.set_xlim(0, width_height[0])
    ax.set_ylim(0, width_height[1])
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Scene camera x [px]")
    ax.set_ylabel("Scene camera y [px]")
    if show:
        plt.show()

    return fig, ax


def overlay_scanpath(
    video: "Video",
    scanpath: pd.DataFrame,
    circle_radius: int = 10,
    line_thickness: int = 2,
    text_size: Optional[int] = None,
    max_fixations: int = 10,
    show_video: bool = False,
    video_output_path: Optional[Path | str] = "scanpath.mp4",
) -> None:
    """
    Plot scanpath on top of the video frames. The resulting video can be displayed and/or saved.

    Parameters
    ----------
    video : SceneVideo
        Video object to overlay the fixations on.
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
    video_output_path : pathlib.Path or str or None
        Path to save the video with fixations overlaid. If None, the video is not saved.
        Defaults to 'scanpath.mp4'.
    """
    # Either show video or save it
    if video_output_path is None and not show_video:
        raise ValueError(
            "Either show_video=True or video_output_path must be provided."
        )

    # Check scanpath DataFrame
    if "fixations" not in scanpath.columns:
        raise ValueError("scanpath DataFrame must contain a 'fixations' column.")
    if not np.allclose(scanpath.index, video.ts):
        raise ValueError("Gaze and video timestamps do not match.")

    # reset video to the beginning
    video.reset()

    # Initialize video capture and writer
    if video_output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_output_path = Path(video_output_path)
        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            video_output_path, fourcc, video.fps, (video.width, video.height)
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
            cv2.imshow("Fixations Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write the frame with overlays to the output video
        if video_output_path is not None:
            out.write(frame)

    # Release resources
    out.release()
    cv2.destroyAllWindows()
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)


@fill_doc
def plot_epochs(
    epochs: "Epochs",
    column_name: Optional[str] = None,
    cmap_name: str = "cool",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot data from a specified column from epochs on a matplotlib axis.

    Works for Epochs created from Stream (time series) or Events (point-based).

    Parameters
    ----------
    epochs : Epochs
        Epochs object containing the data to plot.
    column_name : str
        Name of the column to plot for :class:`pyneon.Epochs` constructed
        from a :class:`pyneon.Stream`. If :class:`pyneon.Epochs` was constructed
        from a :class:`pyneon.Events`, this parameter is ignored. Defaults to None.
    cmap_name : str
        Colormap to use for different epochs. Defaults to 'cool'.
    %(ax_param)s
    %(show_param)s

    Returns
    -------
    %(fig_ax_return)s
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    num_epochs = len(epochs.epochs_dict)
    cmap = cm.get_cmap(cmap_name, num_epochs)
    norm = Normalize(vmin=0, vmax=num_epochs - 1)

    ax.axvline(0, color="k", linestyle="--")

    from pyneon import Events, Stream  # for source class check

    if isinstance(epochs.source, Stream):
        _plot_stream_epochs(epochs.epochs_dict, column_name, cmap, norm, fig, ax)
        ax.set_ylabel(column_name)
    elif isinstance(epochs.source, Events):
        _plot_event_epochs(epochs.epochs_dict, norm, ax)
        ax.set_ylabel("Epoch index")
    else:
        raise ValueError(
            "Epochs must be created from a `Stream` or `Events` to be plotted."
        )

    ax.set_xlabel("Epoch time (s)")
    ax.set_xlim(-epochs.t_before.max() / 1e9, epochs.t_after.max() / 1e9)

    if show:
        plt.show()
    return fig, ax


def _plot_stream_epochs(epochs, column_name, cmap, norm, fig, ax):
    """
    Internal helper to plot Stream-based Epochs.
    """
    if column_name is None:
        raise ValueError("Column name must be provided for Stream-based Epochs.")
    # Iterate through keys and values of the epochs.epochs dictionary
    for i, stream in epochs.items():
        if stream is None or stream.data.empty:
            continue  # Skip bad/missing data
        if column_name not in stream.columns:
            raise ValueError(f"Column '{column_name}' not found in epochs.")
        times = stream["epoch time [ns]"] / 1e9
        data = stream[column_name]
        color = cmap(norm(i))
        ax.plot(times, data, color=color)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch index")


def _plot_event_epochs(epochs, ax):
    """
    Internal helper to plot Events-based Epochs.
    """
    for i, events in epochs.items():
        if events is None:
            continue
        times = events.data["epoch time [ns]"] / 1e9

        # check if duration ms is in df
        if "duration [ms]" in events.data.columns:
            durations = events.data["duration [ms]"] / 1e3
            ax.hlines([i] * len(times), times, times + durations, color="gray")
        else:
            ax.scatter(times, [i] * len(times))


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
    video_output_path = Path(video_output_path)
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

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
            # Resize the frame for display
            resized_frame = cv2.resize(
                frame, (width // 2, height // 2)
            )  # Adjust scaling factor as needed
            cv2.imshow("Video with Overlays", resized_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
