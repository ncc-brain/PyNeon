import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional
from pyneon.stream import Stream

if TYPE_CHECKING:
    from .video import SceneVideo


import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_apriltags(
    video: "SceneVideo",
    tag_family: str = "tag36h11",
    nthreads: int = 4,
    quad_decimate: float = 1.0,
    skip_frames: int = 1,
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Detect AprilTags in a video and report their data for every processed frame,
    optionally using random access instead of sequential reading.

    Parameters
    ----------
    video : SceneVideo
        Scene video to detect AprilTags from.
    tag_family : str, optional
        The AprilTag family to detect (default 'tag36h11').
    nthreads : int, optional
        Number of CPU threads for the detector (default 4).
    quad_decimate : float, optional
        Downsample input frames by this factor for detection (default 1.0).
        Larger values = faster detection, but might miss smaller tags.
    skip_frames : int, optional
        If > 1, detect tags only in every Nth frame.
        E.g., skip_frames=5 will process frames 0, 5, 10, 15, etc.
        Default is 1 (process every frame).
    return_diagnostics : bool, optional
        If True, return additional diagnostic information.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing AprilTag detections, with columns:
        - 'timestamp [ns]' (index): The timestamp of the frame
        - 'processed_frame_idx': The count of processed frames (0-based)
        - 'frame_idx': The actual frame index in the video
        - 'tag_id': The ID of the detected AprilTag
        - 'corners': (4,2) array of tag corner coordinates
        - 'center': (1,2) array for the tag center
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError(
            "To detect AprilTags, the module `pupil-apriltags` is needed. "
            "Install via: pip install pupil-apriltags"
        )

    # Initialize the detector
    detector = Detector(
        families=tag_family,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
    )

    random_access = True
    if skip_frames < 1:
        raise ValueError("skip_frames must be >= 1")
    if skip_frames < 5:
        print(
            "Warning: skip_frames < 5 may be inefficient with random access. Switching to sequential access."
        )
        random_access = False  # sequential access is faster for small skips

    total_frames = len(video.ts)
    all_detections = []
    processed_frame_idx = 0  # counts how many frames we've actually processed

    # -----------------------------------------------------------------------
    # Random-Access Approach
    # -----------------------------------------------------------------------
    if random_access:
        frames_to_process = range(0, total_frames, skip_frames)
        for actual_frame_idx in tqdm(
            frames_to_process, desc="Detecting AprilTags (random access)"
        ):
            # Seek directly to the desired frame
            video.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
            ret, frame = video.read()
            if not ret:
                break

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect
            detections = detector.detect(gray_frame)

            # Save
            for detection in detections:
                corners = detection.corners
                center = np.mean(corners, axis=0)
                results = {
                    "processed_frame_idx": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
                    "timestamp [ns]": video.ts[actual_frame_idx],
                    "tag_id": detection.tag_id,
                    "corners": corners,
                    "center": center,
                }

                if return_diagnostics:
                    results["hamming"] = detection.hamming
                    results["decision_margin"] = detection.decision_margin

                all_detections.append(results)

            processed_frame_idx += 1

    # -----------------------------------------------------------------------
    # Sequential Approach
    # -----------------------------------------------------------------------
    else:
        # We'll just read frames in order, skipping those we don't want
        for actual_frame_idx in tqdm(
            range(total_frames), desc="Detecting AprilTags (sequential)"
        ):
            ret, frame = video.read()
            if not ret:
                break

            if actual_frame_idx % skip_frames != 0:
                continue  # skip

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect
            detections = detector.detect(gray_frame)

            # Save
            for detection in detections:
                corners = detection.corners
                center = np.mean(corners, axis=0)
                results = {
                    "processed_frame_idx": processed_frame_idx,
                    "frame_idx": actual_frame_idx,
                    "timestamp [ns]": video.ts[actual_frame_idx],
                    "tag_id": detection.tag_id,
                    "corners": corners,
                    "center": center,
                }

                if return_diagnostics:
                    results["hamming"] = detection.hamming
                    results["decision_margin"] = detection.decision_margin

                all_detections.append(results)

            processed_frame_idx += 1

    # -----------------------------------------------------------------------
    # Create and return the DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_detections)

    if df.empty:
        return df  # no detections found

    df["method"] = "apriltag"
    df.set_index("timestamp [ns]", inplace=True)
    return df
