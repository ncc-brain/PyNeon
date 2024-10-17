import pandas as pd
import numpy as np
import cv2
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, Optional

from ..preprocess import window_average

if TYPE_CHECKING:
    from ..recording import NeonRecording
    from ..stream import NeonGaze
    from .video import NeonVideo


def sync_gaze_to_video(
    rec: "NeonRecording",
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Synchronize gaze data to video frames by applying windowed averaging
    around each video frame timestamp. See :func:`window_average` for details
    on the averaging process.

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
    sync_gaze = window_average(rec.video.ts, rec.gaze.data, window_size)

    # Mark the fixation status of each frame
    sync_gaze["fixation status"] = pd.Series(dtype="string")

    for fixation_id in sync_gaze["fixation id"].dropna().unique():
        fix_data_index = sync_gaze.loc[sync_gaze["fixation id"] == fixation_id].index
        start_idx = fix_data_index[0]
        end_idx = fix_data_index[-1]
        during_idx = fix_data_index[1:-1]
        sync_gaze.loc[during_idx, "fixation status"] = "during"
        sync_gaze.at[start_idx, "fixation status"] = "start"
        sync_gaze.at[end_idx, "fixation status"] = "end"

    return sync_gaze


def estimate_scanpath(
    video: "NeonVideo",
    sync_gaze: "NeonGaze",
    lk_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Map fixations to video frames using optical flow.

    Parameters
    ----------
    video : NeonVideo
        Video object containing the frames.
    sync_gaze : NeonGaze
        Gaze data synchronized with the video frames.
    lk_params : dict, optional
        Parameters for the Lucas-Kanade optical flow algorithm.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the scanpath with updated fixation points.
    """
    if not np.allclose(sync_gaze.ts, video.ts):
        raise ValueError("Gaze and video timestamps do not match.")
    # Default parameters for Lucas-Kanade optical flow from Neon
    lk_params = (
        {
            "winSize": (90, 90),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        }
        if lk_params is None
        else lk_params
    )
    gaze_data = sync_gaze.data.copy().reset_index(drop=True)

    # Initiate scanpath DataFrame, indexed by video timestamps and containing fixation DataFrames
    scanpath = pd.DataFrame(index=sync_gaze.ts, columns=["fixations"], dtype="object")
    scanpath["fixations"] = [
        gaze_data.loc[
            i, ["fixation id", "gaze x [px]", "gaze y [px]", "fixation status"]
        ]
        .to_frame()
        .T
        for i in gaze_data.index
    ]

    # reset video to the beginning
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_image = None
    for i_frame in tqdm(range(scanpath.shape[0]), desc="Estimating scanpath"):
        # Read the current frame from the video
        ret, frame = video.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i_frame} from the video.")

        curr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i_frame >= 1:
            # Estimate the new fixation points using optical flow for
            # fixations that just ended or are being tracked
            prev_fixations = scanpath.iat[i_frame - 1, 0].copy()
            prev_fixations = prev_fixations[
                (prev_fixations["fixation status"] == "end")
                | (prev_fixations["fixation status"] == "tracked")
            ]

            if not prev_fixations.empty:
                # Prepare x y points for optical flow
                prev_pts = (
                    prev_fixations[["gaze x [px]", "gaze y [px]"]]
                    .to_numpy()
                    .astype(np.float32)
                    .reshape(-1, 1, 2)
                )
                prev_ids = prev_fixations["fixation id"].values

                # Calculate optical flow to find new positions of the points
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_image, curr_image, prev_pts, None, **lk_params
                )

                # Update fixations for the current frame
                # First copy the online fixation status for this frame
                curr_fixations = scanpath.iat[i_frame, 0]

                # Append tracked fixation points
                for i, (pt, s) in enumerate(zip(curr_pts, status)):
                    if s[0]:  # Check if the point was successfully tracked
                        x, y = pt.ravel()
                        fixation = pd.DataFrame(
                            {
                                "fixation id": prev_ids[i],
                                "gaze x [px]": x,
                                "gaze y [px]": y,
                                "fixation status": "tracked",
                            },
                            index=[prev_ids[i]],
                        )
                    else:  # No longer trackable, denote as lost
                        fixation = pd.DataFrame(
                            {
                                "fixation id": prev_ids[i],
                                "gaze x [px]": None,
                                "gaze y [px]": None,
                                "fixation status": "lost",
                            },
                            index=[prev_ids[i]],
                        )
                    curr_fixations = pd.concat(
                        [curr_fixations, fixation], ignore_index=True
                    )

                # Update the DataFrame with the modified fixations
                scanpath.iat[i_frame, 0] = curr_fixations

        # Update the previous frame for the next iteration
        prev_image = curr_image

    # Reset video to the beginning
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return scanpath
