import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from ..stream import Gaze
    from .video import SceneVideo


def estimate_scanpath(
    video: "SceneVideo",
    sync_gaze: "Gaze",
    lk_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Track fixations from frame to frame with Lucas‑Kanade optical flow.

    Parameters
    ----------
    video : SceneVideo
        Video object (already opened with OpenCV) that provides frames
        and the vector ``video.ts`` (int64 ns) with one timestamp per frame.
    sync_gaze : NeonGaze
        Gaze stream down‑sampled to the same timestamps as the video
        (``sync_gaze.ts`` == ``video.ts``). Must contain columns
        ``"fixation id"``, ``"gaze x [px]"``, ``"gaze y [px]"`` and the
        categorical ``"fixation status"`` (values “start”, “during”, “end”).
    lk_params : dict, optional
        Parameters forwarded to :pyfunc:`cv2.calcOpticalFlowPyrLK`.
        If *None*, reasonable defaults are used.

    Returns
    -------
    pandas.DataFrame
        One row per video frame, indexed by ``"timestamp [ns]"`` and
        containing a single column ``"fixations"``.  Each cell holds a
        **nested** DataFrame with the columns:

        * ``fixation id``   (int)
        * ``gaze x [px]``   (float or NaN)
        * ``gaze y [px]``   (float or NaN)
        * ``fixation status`` (“start”, “during”, “tracked”, “lost”)
    """
    # ------------------------------------------------------------------ checks
    if not np.allclose(sync_gaze.ts, video.ts):
        raise ValueError("Gaze and video timestamps do not match.")

    lk_params = lk_params or {
        "winSize": (90, 90),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    }

    gaze_df = sync_gaze.data.copy().reset_index(drop=True)

    # ------------------------------------------------------------------ build initial status map per fixation
    gaze_df["fixation status"] = pd.NA
    for fid in gaze_df["fixation id"].dropna().unique():
        idx = gaze_df.index[gaze_df["fixation id"] == fid]
        if len(idx) == 0:
            continue
        gaze_df.loc[idx[0], "fixation status"] = "start"
        gaze_df.loc[idx[-1], "fixation status"] = "end"
        if len(idx) > 2:
            gaze_df.loc[idx[1:-1], "fixation status"] = "during"

    # ------------------------------------------------------------------ containers
    active: dict[int, tuple[float, float]] = {}  # currently trackable (x, y)
    frames: list[dict[int, tuple[float, float, str]]] = []

    # first frame: fill from gaze_df row 0
    first_row = gaze_df.iloc[0]
    if not pd.isna(first_row["fixation id"]):
        fid = int(first_row["fixation id"])
        xy = (float(first_row["gaze x [px]"]), float(first_row["gaze y [px]"]))
        status = first_row["fixation status"]
        frames.append({fid: (*xy, status)})
        if status in {"start", "during"}:
            active[fid] = xy
    else:
        frames.append({})

    # ------------------------------------------------------------------ LK loop
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_bgr = video.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(1, len(sync_gaze.ts)), desc="Estimating scanpath"):
        ret, curr_bgr = video.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {i}.")

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        frame_state: dict[int, tuple[float, float, str]] = {}

        # ---------- propagate existing points ---------------------------------
        if active:
            prev_pts = np.array(list(active.values()), dtype=np.float32).reshape(
                -1, 1, 2
            )
            prev_ids = np.array(list(active.keys()), dtype=np.int32)

            curr_pts, status_mask, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )

            for (x, y), ok, fid in zip(
                curr_pts.reshape(-1, 2), status_mask.ravel(), prev_ids
            ):
                if ok:
                    frame_state[fid] = (float(x), float(y), "tracked")
                    active[fid] = (float(x), float(y))
                else:
                    frame_state[fid] = (np.nan, np.nan, "lost")
                    active.pop(fid, None)

        # ---------- add new fixations starting at this frame ------------------
        row = gaze_df.iloc[i]
        if not pd.isna(row["fixation id"]):
            fid_new = int(row["fixation id"])
            if fid_new not in frame_state:  # not already propagated
                xy_new = (row["gaze x [px]"], row["gaze y [px]"])
                status_new = row["fixation status"]
                frame_state[fid_new] = (*xy_new, status_new)
                if status_new in {"start", "during"}:
                    active[fid_new] = (float(xy_new[0]), float(xy_new[1]))

        frames.append(frame_state)
        prev_gray = curr_gray

    # ------------------------------------------------------------------ build final DataFrame
    records = []
    for ts, state in zip(sync_gaze.ts, frames):
        if state:
            fix_df = (
                pd.DataFrame.from_dict(
                    state,
                    orient="index",
                    columns=["gaze x [px]", "gaze y [px]", "fixation status"],
                )
                .assign(**{"fixation id": lambda d: d.index})
                .reset_index(drop=True)
            )
        else:
            fix_df = pd.DataFrame(
                columns=["fixation id", "gaze x [px]", "gaze y [px]", "fixation status"]
            )
        records.append((ts, fix_df))

    scanpath = pd.DataFrame.from_records(
        records, columns=["timestamp [ns]", "fixations"]
    ).set_index("timestamp [ns]")
    scanpath.index.name = "timestamp [ns]"

    # reset video so other routines can reuse it
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return scanpath
