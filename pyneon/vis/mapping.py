import cv2
import numpy as np
import random
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..video import SceneVideo

def show_random_frame_detections(
    video: "SceneVideo",
    detection_df: pd.DataFrame,
    show_ids: bool = True,
    thickness: int = 2,
    wait_key: int = 0,
):
    """
    Display a random frame from the video with all detected polygons (tags/screens) drawn.

    Parameters
    ----------
    video : SceneVideo
        Video object supporting OpenCV-like `set()` and `read()` methods.
    detection_df : pandas.DataFrame
        DataFrame with at least:
            - 'frame_idx': int
            - 'corners': ndarray (4, 2)
            - optional: 'tag_id' (int)
            - optional: 'method' (str)
    show_ids : bool, optional
        Whether to draw tag IDs or indices near polygons (default: True).
    thickness : int, optional
        Line thickness for polygons (default: 2).
    wait_key : int, optional
        Milliseconds to wait in `cv2.waitKey()`. Default 0 = wait for key press.
    """

    if detection_df.empty:
        print("No detections available.")
        return

    # Pick a random frame that has detections
    frame_idx = random.choice(detection_df["frame_idx"].unique())
    dets = detection_df[detection_df["frame_idx"] == frame_idx]
    print(f"Displaying frame {frame_idx} with {len(dets)} detections.")

    # Retrieve frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        return

    # Generate distinct colors for multiple detections
    rng = np.random.default_rng(seed=42)
    colors = (rng.random((len(dets), 3)) * 255).astype(np.uint8)

    for i, (_, det) in enumerate(dets.iterrows()):
        corners = np.array(det["corners"], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(c) for c in colors[i])
        tag_id = det.get("tag_id", i)
        method = det.get("method", "")

        # Draw polygon outline
        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=thickness)

        # Draw corner points
        for j, (x, y) in enumerate(corners.reshape(-1, 2)):
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        # Label the detection
        if show_ids:
            center = np.mean(corners.reshape(-1, 2), axis=0)
            label = f"ID {tag_id}"
            if method:
                label += f" ({method})"
            cv2.putText(
                frame,
                label,
                (int(center[0]) - 30, int(center[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    # Display
    cv2.imshow(f"Detections in frame {frame_idx}", frame)
    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()