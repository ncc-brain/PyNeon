import pandas as pd
import numpy as np
import cv2
import warnings
from pathlib import Path

from typing import TYPE_CHECKING, Union, Optional

from ..preprocess import window_average

if TYPE_CHECKING:
    from ..recording import NeonRecording


def sync_gaze_to_video(
    rec: "NeonRecording",
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Synchronize gaze data to video frames by applying windowed averaging
    around each video frame timestamp.

    Parameters:
    -----------
    rec : NeonRecording
        Recording object containing gaze and video data.
    window_size : int, optional
        The size of the time window (in nanoseconds)
        over which to compute the average around each new timestamp.
        If ``None`` (default), the window size is set to the median interval
        between the new timestamps, i.e., ``np.median(np.diff(new_ts))``.
        The window size must be larger than the median interval between the original data timestamps,
        i.e., ``window_size > np.median(np.diff(data.index))``.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing gaze data mapped to video timestamps.
    """
    gaze = rec.gaze
    video = rec.video

    if gaze is None:
        raise ValueError("No gaze data available.")
    if video is None:
        raise ValueError("No video data available.")

    # Resample the gaze data to the video timestamps
    mapped_gaze = window_average(rec.video.ts, rec.gaze.data, window_size)

    # Mark the fixation status of each frame
    mapped_gaze["fixation status"] = pd.Series(dtype="string")

    for fixation_id in mapped_gaze["fixation id"].dropna().unique():
        current_gaze_data = mapped_gaze.loc[mapped_gaze["fixation id"] == fixation_id]
        start_idx = current_gaze_data.index[0]
        end_idx = current_gaze_data.index[-1]
        during_idx = current_gaze_data.index[1:-1]
        mapped_gaze.loc[during_idx, "fixation status"] = "during"
        mapped_gaze.at[start_idx, "fixation status"] = "start"
        mapped_gaze.at[end_idx, "fixation status"] = "end"

    return mapped_gaze


def estimate_scanpath(
    rec: "NeonRecording",
    lk_params: Union[None, dict] = None,
) -> pd.DataFrame:
    """
    Map fixations to video frames.

    Parameters
    ----------
    rec : NeonRecording
        Recording object containing gaze and video data.
    lk_params : dict
        Parameters for the Lucas-Kanade optical flow algorithm.
    """

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # check if rec.mapped_gaze exists, if not, create it
    if not hasattr(rec, "mapped_gaze") or rec.mapped_gaze is None:
        mapped_gaze = sync_gaze_to_video(rec)
    else:
        mapped_gaze = rec.mapped_gaze

    # create a new dataframe of dataframes that stores the relevant data for each fixation
    mapped_gaze.rename(
        columns={"time [s]": "time", "gaze x [px]": "x", "gaze y [px]": "y"},
        inplace=True,
    )
    estimated_scanpath = pd.DataFrame(columns=["time", "fixations"])

    for idx in range(mapped_gaze.shape[0]):
        estimated_scanpath = estimated_scanpath._append(
            {
                "time": mapped_gaze.loc[idx, "time"],
                "fixations": mapped_gaze.loc[
                    [idx], ["fixation id", "x", "y", "fixation status"]
                ],
            },
            ignore_index=True,
        )

    video = rec.video
    # reset video to the beginning
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Taken from Neon
    if lk_params is None:
        lk_params = dict(
            winSize=(90, 90),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    prev_frame = None
    prev_fix = None

    for idx in estimated_scanpath.index:
        # Read the current frame from the video
        ret, frame = video.read()
        if not ret:
            break

        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if idx >= 1:
            # Access previous frame fixations and filter by status
            prev_fixations = estimated_scanpath.at[idx - 1, "fixations"]
            prev_fixations = prev_fixations[
                (prev_fixations["fixation status"] == "end")
                | (prev_fixations["fixation status"] == "tracked")
            ]

            if not prev_fixations.empty:
                # Prepare points for tracking
                prev_pts = np.array(
                    prev_fixations[["x", "y"]].dropna().values, dtype=np.float32
                ).reshape(-1, 1, 2)
                prev_ids = prev_fixations["fixation id"].values

                # Calculate optical flow to find new positions of the points
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_frame, curr_frame, prev_pts, None, **lk_params
                )

                # Update fixations for the current frame
                curr_fixations = estimated_scanpath.at[idx, "fixations"].copy()

                # Append new or updated fixation points
                for i, (pt, s, e) in enumerate(zip(curr_pts, status, err)):
                    if s[0]:  # Check if the point was successfully tracked
                        x, y = pt.ravel()
                        curr_fixations = curr_fixations._append(
                            {
                                "fixation id": prev_ids[i],
                                "x": x,
                                "y": y,
                                "fixation status": "tracked",
                            },
                            ignore_index=True,
                        )
                    else:
                        # Handle cases where the point could not be tracked
                        curr_fixations = curr_fixations._append(
                            {
                                "fixation id": prev_ids[i],
                                "x": None,
                                "y": None,
                                "fixation status": "lost",
                            },
                            ignore_index=True,
                        )

                # Update the DataFrame with the modified fixations
                estimated_scanpath.at[idx, "fixations"] = curr_fixations

        # Update the previous frame for the next iteration
        prev_frame = curr_frame

    rec.scanpath = estimated_scanpath

    return estimated_scanpath


def overlay_scanpath_on_video(
    rec: "NeonRecording",
    video_output_path: Union[Path, str] = "sacnpath_overlay_video.mp4",
    circle_radius: int = 10,
    show_lines: bool = True,
    line_thickness: int = 2,
    show_video: bool = False,
    max_fixations: int = 10,
) -> None:
    """
    Overlay fixations and gaze data on video frames and save the resulting video.

    Parameters
    ----------
    rec : NeonRecording
        Recording object containing gaze and video data.
    video_output_path : str
        Path where the video with fixations will be saved.
    circle_radius : int
        Radius of the circle used to represent fixations.
    line_thickness : int
        Thickness of the lines connecting successive fixations.
    show_video : bool
        Flag to display the video with fixations overlaid in
    """
    video_output_path = Path(video_output_path)
    video_output_path.parent.mkdir(parents=True, exist_ok=True)

    warnings.simplefilter(action="ignore", category=FutureWarning)

    if rec.video:
        video = rec.video
    else:
        raise ValueError("No video data found.")

    if not hasattr(rec, "estimated_scanpath") or rec.estimated_scanpath is None:
        print("Estimating scanpath...")
        df = estimate_scanpath(rec)
    else:
        df = rec.estimated_scanpath

    # reset video to the beginning
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize video capture and writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        video_output_path, fourcc, video.fps, (video.width, video.height)
    )

    # Iterate through each row in the tracked fixations DataFrame
    prev_x, prev_y = None, None  # Track previous fixation point for drawing lines
    for idx, row in df.iterrows():
        # Read the current frame from the video
        ret, frame = video.read()
        if not ret:
            print("End of video")
            break

        # Extract fixations and gaze data
        fixations = row["fixations"]
        prev_x, prev_y = None, None

        # check if fixations is empty
        if fixations.empty:
            continue

        for i in range(len(fixations)):
            fixation_x, fixation_y = fixations.iloc[i]["x"], fixations.iloc[i]["y"]
            status = fixations.iloc[i]["fixation status"]
            id = fixations.iloc[i]["fixation id"]

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
                if pd.notna(fixation_x) and pd.notna(fixation_y):
                    cv2.circle(
                        frame,
                        (int(fixation_x), int(fixation_y)),
                        radius=circle_radius,
                        color=color,
                        thickness=-1,
                    )

                    # Optionally add text showing fixation status and ID
                    cv2.putText(
                        frame,
                        f"ID: {id} Status: {status}",
                        (int(fixation_x) + 10, int(fixation_y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        1,
                    )

                    # Draw line connecting the previous fixation to the current one
                    if pd.notna(prev_x) and pd.notna(prev_y) and show_lines:
                        cv2.line(
                            frame,
                            (int(prev_x), int(prev_y)),
                            (int(fixation_x), int(fixation_y)),
                            color,
                            line_thickness,
                        )

                    # Update the previous fixation point
                    prev_x, prev_y = fixation_x, fixation_y

        # Display the frame with overlays (Optional)
        if show_video:
            cv2.imshow("Fixations Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write the frame with overlays to the output video
        out.write(frame)

    # Release resources
    out.release()
    cv2.destroyAllWindows()
