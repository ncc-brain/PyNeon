import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Optional, TYPE_CHECKING, Tuple, Any, Dict, Union
from pathlib import Path

from tqdm import tqdm

if TYPE_CHECKING:
    from ..video import SceneVideo

def plot_frame_detections(
    video: "SceneVideo",
    detection_df: pd.DataFrame,
    frame_index: Optional[int] = None,
    show_ids: bool = True,
    thickness: int = 2,
    figsize: tuple = (12, 7),
    ax: Optional[plt.Axes] = None,
):
    """
    Show a frame with all detected polygons (tags/screens) overlaid using matplotlib.

    Parameters
    ----------
    video : SceneVideo
        Video object supporting OpenCV-like `set()` and `read()` methods, and `ts`.
    detection_df : pandas.DataFrame
        Must contain:
            - 'frame_idx': int
            - 'corners' : ndarray (4, 2)
        Optional:
            - 'tag_id': int
            - 'method': str
    frame_index : int, optional
        Frame index to display. If None, a random frame that has detections is chosen.
    show_ids : bool, optional
        Draw 'tag_id' and (optional) 'method' next to each polygon (default True).
    thickness : int, optional
        Polygon edge thickness (matplotlib linewidth). Default 2.
    figsize : tuple, optional
        Figure size in inches. Default (12, 7).
    ax : matplotlib.axes.Axes, optional
        An existing axes to draw on. If None, a new figure/axes is created.
    """
    if detection_df.empty:
        print("No detections available.")
        return

    # Select frame
    frames_with_dets = detection_df["frame_idx"].unique()
    if frame_index is None:
        frame_index = int(random.choice(frames_with_dets))
    elif frame_index not in frames_with_dets:
        print(f"Frame {frame_index} has no detections; available frames: {frames_with_dets[:10]} ...")
        return

    dets = detection_df[detection_df["frame_idx"] == frame_index]
    # Load frame
    video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame_bgr = video.read()
    if not ret:
        print(f"Failed to read frame {frame_index}")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Prepare axes
    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(frame_rgb)
    ax.set_title(f"Detections in frame {frame_index} (n={len(dets)})")
    ax.axis("off")

    # Color cycle
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(dets))]

    for i, (_, det) in enumerate(dets.iterrows()):
        corners = np.asarray(det["corners"], dtype=np.float32).reshape(4, 2)
        poly = Polygon(corners, closed=True, fill=False, edgecolor=colors[i], linewidth=thickness)
        ax.add_patch(poly)

        # corner dots
        ax.scatter(corners[:, 0], corners[:, 1], s=24, c=[colors[i]]*4)

        if show_ids:
            tag_id = det.get("tag_id", i)
            method = det.get("method", None)
            center = corners.mean(axis=0)
            label = f"ID {tag_id}" + (f" ({method})" if method else "")
            ax.text(center[0] + 8, center[1] - 8, label,
                    color=colors[i], fontsize=10, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.4), ec="none"))

    if created_fig:
        plt.tight_layout()
        plt.show()


def detection_overlay_video(
    video: "SceneVideo",
    detection_df: pd.DataFrame,
    out_path: Union[str, Path],
    fps: Optional[float] = None,
    max_miss: int = 30,
    thickness: int = 2,
    draw_ids: bool = True,
    codec: str = "mp4v",
    frame_range: Optional[Tuple[int, int]] = None,
):
    """
    Create an overlay video with detection polygons. Polygons are green when detected;
    if a tag disappears, the last known polygon is kept and its color fades towards red
    over `max_miss` frames, after which it stops being drawn.

    Parameters
    ----------
    video : SceneVideo
        Video object supporting OpenCV-like `set()` and `read()` methods, and `ts`.
    detection_df : pandas.DataFrame
        Must contain:
            - 'frame_idx' : int
            - 'corners'   : ndarray (4, 2)
        Optional:
            - 'tag_id'    : int (default groups by unique polygon; if missing, uses row index)
            - 'method'    : str
    out_path : str
        Output video path (e.g., "overlaid.mp4").
    fps : float, optional
        Frames per second for the output. If None, tries `video.fps`, else derives from timestamps,
        else defaults to 30.
    max_miss : int, optional
        Number of frames to keep and fade a missing detection before dropping. Default 60.
    thickness : int, optional
        Line thickness for polygons. Default 2.
    draw_ids : bool, optional
        Draw tag IDs and methods as labels. Default True.
    codec : str, optional
        FourCC codec for OpenCV VideoWriter. Default "mp4v".
    frame_range : (start, end), optional
        Process only frames in [start, end) (end exclusive). Defaults to all frames.

    Returns
    -------
    str
        The output path written.
    """
    if detection_df.empty:
        raise ValueError("detection_df is empty; nothing to overlay.")

    # Determine FPS
    if fps is None:
        fps = getattr(video, "fps", None)
    if fps is None:
        # derive from timestamps if possible
        if hasattr(video, "ts") and len(video.ts) > 1:
            dt_ns = np.median(np.diff(np.asarray(video.ts, dtype=np.int64)))
            if dt_ns > 0:
                fps = float(1e9 / dt_ns)
    if fps is None:
        fps = 30.0

    # Frame size from first frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, fr0 = video.read()
    if not ok:
        raise RuntimeError("Cannot read first frame to determine frame size.")
    height, width = fr0.shape[:2]

    # Build frame index range
    nframes_total = len(getattr(video, "ts", [])) or int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start = 0 if frame_range is None else int(frame_range[0])
    end = nframes_total if frame_range is None else int(frame_range[1])
    start = max(0, start)
    end = min(nframes_total, end)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for '{out_path}'.")

    # Group detections by frame
    dets_by_frame: Dict[int, list] = {}
    for _, row in detection_df.iterrows():
        f = int(row["frame_idx"])
        dets_by_frame.setdefault(f, []).append(row)

    # Track last seen state per tag_id
    # state[tag_id] = {"last_frame": int, "corners": np.ndarray (4,2), "method": str}
    state: Dict[Any, Dict[str, Any]] = {}

    # Position reader and iterate sequentially
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in tqdm(range(start, end), desc="Creating detection overlay video"):
        ok, frame = video.read()
        if not ok:
            break

        # 1) Draw current detections in GREEN and refresh state
        current = dets_by_frame.get(f, [])
        seen_ids = set()
        for det in current:
            corners = np.asarray(det["corners"], dtype=np.float32).reshape(4, 2)
            tag_id = det.get("tag_id", None)
            if tag_id is None:
                # stable id if not supplied: use tuple(corners.ravel()) – but here
                # just use the row id to keep it simple:
                tag_id = hash(tuple(corners.ravel()))
            method = det.get("method", "")

            # Draw polygon (green)
            pts = corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), thickness)
            if draw_ids:
                cxy = corners.mean(axis=0).astype(int)
                label = f"{tag_id}" + (f" ({method})" if method else "")
                cv2.putText(frame, label, (cxy[0] + 8, cxy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # update state
            state[tag_id] = {"last_frame": f, "corners": corners, "method": method}
            seen_ids.add(tag_id)

        # 2) Draw faded (missing) detections up to max_miss frames
        for tag_id, info in list(state.items()):
            if tag_id in seen_ids:
                continue  # already drawn green
            age = f - int(info["last_frame"])
            if age <= 0 or age > max_miss:
                # too old: drop
                if age > max_miss:
                    state.pop(tag_id, None)
                continue

            # interpolate GREEN -> RED by age/max_miss
            t = min(1.0, age / float(max_miss))
            g = int(round(255 * (1.0 - t)))
            r = int(round(255 * t))
            color = (0, g, r)  # BGR

            corners = info["corners"]
            pts = corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(frame, [pts], True, color, thickness)
            if draw_ids:
                cxy = corners.mean(axis=0).astype(int)
                label = f"{tag_id} (age {age})"
                if info.get("method"):
                    label = f"{tag_id}, {info['method']}, (age {age})"
                cv2.putText(frame, label, (cxy[0] + 8, cxy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    video.reset()
    return out_path
