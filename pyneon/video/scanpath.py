from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from ..stream import Stream
    from .video import Video


def estimate_scanpath(
    video: "Video",
    sync_gaze: "Stream",
    lk_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Map fixations to video frames using optical flow.

    Parameters
    ----------
    video : Video
        Video object containing the frames.
    sync_gaze : Stream
        Gaze data synchronized with the video frames.
    lk_params : dict, optional
        Parameters for the Lucas-Kanade optical flow algorithm.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the scanpath with updated fixation points.
    """
    if not np.allclose(sync_gaze.ts, video.ts):
        raise ValueError("Gaze and video timestamps do not match.")

    # Default parameters for Lucas-Kanade optical flow from Pupil Labs
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

    # Mark the fixation status of each frame
    gaze_data["fixation status"] = pd.Series(dtype="string")

    for fixation_id in gaze_data["fixation id"].dropna().unique():
        fix_data_index = gaze_data.loc[gaze_data["fixation id"] == fixation_id].index
        start_idx = fix_data_index[0]
        end_idx = fix_data_index[-1]
        during_idx = fix_data_index[1:-1]
        gaze_data.loc[during_idx, "fixation status"] = "during"
        gaze_data.at[start_idx, "fixation status"] = "start"
        gaze_data.at[end_idx, "fixation status"] = "end"

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
    video.reset()
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
    video.reset()

    scanpath.index.name = "timestamp [ns]"

    scanpath["frame index"] = np.arange(scanpath.shape[0])

    return scanpath
