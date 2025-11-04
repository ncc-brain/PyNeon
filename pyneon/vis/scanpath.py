import numpy as np
import pandas as pd
import cv2

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from tqdm import tqdm

if TYPE_CHECKING:
    from ..video import SceneVideo


def overlay_scanpath(
    video: "SceneVideo",
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
