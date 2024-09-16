import pandas as pd
import numpy as np
import cv2

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..recording import NeonRecording


def map_gaze_to_video(
    rec: "NeonRecording",
    resamp_float_kind: str = "linear",
    resamp_other_kind: str = "nearest",
) -> pd.DataFrame:
    """
    Map gaze data to video frames.
    """
    gaze = rec.gaze
    video = rec.video

    if gaze is None:
        raise ValueError("No gaze data available.")
    if video is None:
        raise ValueError("No video data available.")

    # Resample the gaze data to the video timestamps
    mapped_gaze = gaze.resample(
        video.ts, float_kind=resamp_float_kind, other_kind=resamp_other_kind
    )

    # Mark the fixation status of each frame
    mapped_gaze["fixation status"] = pd.NA

    for fixation_id in mapped_gaze["fixation id"].dropna().unique():
        current_gaze_data = mapped_gaze.loc[mapped_gaze["fixation id"] == fixation_id]
        start_idx = current_gaze_data.index[0]
        end_idx = current_gaze_data.index[-1]
        during_idx = current_gaze_data.index[1:-1]
        mapped_gaze.loc[during_idx, "fixation status"] = "during"
        mapped_gaze.at[start_idx, "fixation status"] = "start"
        # Assign end last to enforce that the last frame is always marked as end
        mapped_gaze.at[end_idx, "fixation status"] = "end"

    return mapped_gaze


def estimate_past_fixations(
    rec: "NeonRecording",
    lk_params: Union[None, dict] = None,
) -> pd.DataFrame:
    """
    Map fixations to video frames.
    """
    mapped_gaze = map_gaze_to_video(rec)
    # create a new dataframe to store the relevant fixations
    tracked_fixations = pd.DataFrame(columns=["ts", "fixations"])
    for idx in range(mapped_gaze.shape[0]):
        fixations = pd.todf
    video = rec.video

    # Taken from Neon
    if lk_params is None:
        lk_params = dict(
            winSize=(90, 90),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    ret, frame = video.read()
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for idx in range(1, mapped_gaze.shape[0]):
        ret, frame = video.read()
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_fixations = mapped_gaze.loc[idx - 1, "fixations on screen"]
        prev_fixations = prev_fixations[
            (prev_fixations["fixation status"] == "end")
            | (prev_fixations["fixation status"] == "tracked")
        ]
