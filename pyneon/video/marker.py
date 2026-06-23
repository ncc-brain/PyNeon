import re
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..stream import Stream
from ..utils import _validate_df_columns
from ..utils.docstring_templating import fill_doc
from .utils import distort_points, resolve_processing_window
from .variables import (
    APRILTAG_FAMILIES,
    ARUCO_NUMBERS,
    ARUCO_SIZES,
    MARKER_DETECTION_COLUMNS,
)

if TYPE_CHECKING:
    from .video import Video


DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)
DEFAULT_HIGHLIGHT_PERCENTILE = 99.5
DEFAULT_GAUSSIAN_BLUR_SIGMA = 0.8
DEFAULT_SHARPEN_AMOUNT = 1.0

# Keep this in sync with preprocess_marker_frame defaults.
DEFAULT_PREPROCESS_PARAMS = {
    "clahe_clip_limit": DEFAULT_CLAHE_CLIP_LIMIT,
    "clahe_tile_grid_size": DEFAULT_CLAHE_TILE_GRID_SIZE,
    "highlight_percentile": DEFAULT_HIGHLIGHT_PERCENTILE,
    "gaussian_blur_sigma": DEFAULT_GAUSSIAN_BLUR_SIGMA,
    "sharpen_amount": DEFAULT_SHARPEN_AMOUNT,
}

#: Built-in preprocessing presets for :func:`detect_markers`.
#:
#: Each preset is a dict of keyword arguments forwarded to
#: :func:`preprocess_marker_frame`. Pass the preset name as
#: ``preprocess="mild"`` (etc.) to :func:`detect_markers`.
PREPROCESS_PRESETS: dict[str, dict] = {
    "mild": dict(DEFAULT_PREPROCESS_PARAMS),
    "strong_highlight_clipping": {
        "clahe_clip_limit": DEFAULT_CLAHE_CLIP_LIMIT,
        "clahe_tile_grid_size": DEFAULT_CLAHE_TILE_GRID_SIZE,
        "highlight_percentile": 99.0,
        "gaussian_blur_sigma": 1.0,
        "sharpen_amount": 0.6,
    },
    "strong_local_contrast": {
        "clahe_clip_limit": 2.5,
        "clahe_tile_grid_size": DEFAULT_CLAHE_TILE_GRID_SIZE,
        "highlight_percentile": None,
        "gaussian_blur_sigma": 0.6,
        "sharpen_amount": DEFAULT_SHARPEN_AMOUNT,
    },
}


def preprocess_marker_frame(
    gray_frame: np.ndarray,
    *,
    clahe_clip_limit: float | None = 2.0,
    clahe_tile_grid_size: tuple[int, int] | None = (8, 8),
    highlight_percentile: float | None = 99.5,
    gaussian_blur_sigma: float | None = 0.8,
    sharpen_amount: float | None = 1.0,
) -> np.ndarray:
    """Preprocess a grayscale frame to improve AprilTag / ArUco detection.

    Applies a lightweight pipeline intended to help with low-contrast,
    unevenly illuminated, or IR-contaminated scenes. All operations use
    OpenCV/NumPy only and return a ``uint8`` grayscale image compatible
    with :func:`cv2.aruco.ArucoDetector.detectMarkers`.

    The processing order is:

    1. Highlight clipping / compression (optional)
    2. Local contrast enhancement via CLAHE (optional)
    3. Mild Gaussian smoothing (optional, ``gaussian_blur_sigma > 0``)
    4. Unsharp mask sharpening (optional)

    Parameters
    ----------
    gray_frame : numpy.ndarray
        Grayscale input image. ``uint8`` is expected; other numeric dtypes are
        accepted and will be converted to ``uint8`` before processing. Float
        arrays are clipped to ``[0, 255]`` before conversion to prevent
        wrap-around truncation for out-of-range values.
    clahe_clip_limit : float or None, optional
        Clip limit for CLAHE. Higher values give stronger enhancement but
        more noise amplification. CLAHE runs only when both
        ``clahe_clip_limit`` and ``clahe_tile_grid_size`` are not ``None``.
        Set either parameter to ``None`` to skip the CLAHE step. Defaults to
        ``2.0``.
    clahe_tile_grid_size : tuple[int, int] or None, optional
        Tile grid size for CLAHE. CLAHE runs only when both
        ``clahe_clip_limit`` and ``clahe_tile_grid_size`` are not ``None``.
        Set either parameter to ``None`` to skip the CLAHE step. Defaults to
        ``(8, 8)``.
    highlight_percentile : float or None, optional
        Upper percentile used to compress very bright highlights before
        renormalizing to the full 0–255 range. Set to ``None`` to skip
        highlight clipping. Defaults to ``99.5``.
    gaussian_blur_sigma : float or None, optional
        Standard deviation for mild Gaussian smoothing applied before
        sharpening. Set to ``None`` or ``0`` to skip smoothing.
        Defaults to ``0.8``.
    sharpen_amount : float or None, optional
        Strength of the unsharp mask (higher = more sharpening). Set to
        ``None`` to skip sharpening. Defaults to ``1.0``.

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale ``uint8`` image of the same spatial size
        as ``gray_frame``.
    """
    if gray_frame.dtype != np.uint8:
        # For float inputs, clip to [0, 255] before converting to avoid
        # wrap-around truncation for out-of-range values.
        if np.issubdtype(gray_frame.dtype, np.floating):
            gray_frame = np.clip(gray_frame, 0, 255)
        gray_frame = gray_frame.astype(np.uint8)

    img = gray_frame.copy()

    if highlight_percentile is not None:
        high = np.percentile(img, highlight_percentile)
        # Guard against a degenerate all-black image where the percentile
        # would be 0, which would cause cv2.normalize to divide by zero.
        high = max(float(high), 1.0)
        img = np.clip(img, 0, high)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if clahe_clip_limit is not None and clahe_tile_grid_size is not None:
        clahe_op = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )
        img = clahe_op.apply(img)

    blurred = img
    if gaussian_blur_sigma is not None and gaussian_blur_sigma > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), gaussian_blur_sigma)

    if sharpen_amount is not None:
        img_f = img.astype(np.float32)
        blurred_f = blurred.astype(np.float32)
        sharp = img_f + sharpen_amount * (img_f - blurred_f)
        img = np.clip(sharp, 0, 255).astype(np.uint8)
    else:
        img = blurred

    return img


def marker_family_to_dict(marker_family: str) -> Tuple[str, cv2.aruco.Dictionary]:
    # AprilTags
    if marker_family in APRILTAG_FAMILIES:
        dict_name: str = f"DICT_APRILTAG_{marker_family.upper()}"
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        return "april", aruco_dict

    # ArUco Original
    if marker_family.lower() == "aruco_original":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        return "aruco", aruco_dict

    # Other ArUco (format: {size}_{number})
    aruco_pattern = re.compile(r"^(\d+)x\1_(\d+)$")
    pattern_match = aruco_pattern.match(marker_family)

    if pattern_match:
        # Split marker name into size and number components
        size, number = marker_family.split("_")

        if size not in ARUCO_SIZES:
            raise ValueError(
                f"Invalid Aruco marker size '{size}' in '{marker_family}'. "
                f"Supported sizes: {', '.join(ARUCO_SIZES)}"
            )

        if number not in ARUCO_NUMBERS:
            raise ValueError(
                f"Invalid Aruco marker number '{number}' in '{marker_family}'. "
                f"Supported numbers: {', '.join(ARUCO_NUMBERS)}"
            )

        dict_name = f"DICT_{marker_family.upper()}"
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        return "aruco", aruco_dict

    # Provide helpful error message with supported formats
    raise ValueError(
        f"Unrecognized marker family '{marker_family}'. "
        f"Expected format:\n"
        f"  - AprilTag: {', '.join(APRILTAG_FAMILIES)}\n"
        f"  - Aruco: {{size}}_{{number}} (e.g., '6x6_250')\n"
        f"    Available sizes: {', '.join(ARUCO_SIZES)}\n"
        f"    Available numbers: {', '.join(ARUCO_NUMBERS)}"
    )


def generate_marker(
    marker_family: str,
    marker_id: int,
    marker_size_pixels: int,
):
    aruco_dict = marker_family_to_dict(marker_family)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    return img


@fill_doc
def detect_markers(
    video: "Video",
    marker_family: str | list[str],
    step: int = 1,
    processing_window: Optional[tuple[int | float, int | float]] = None,
    processing_window_unit: Literal["frame", "time", "timestamp"] = "frame",
    detector_parameters: Optional[cv2.aruco.DetectorParameters] = None,
    undistort: bool = False,
    preprocess: str | None = "mild",
    preprocess_params: dict | None = None,
) -> Stream:
    """
    Detect fiducial markers (AprilTag or ArUco) in a video and report their data for every processed frame.

    Parameters
    ----------
    video : Video
        Scene video to detect markers from.
    {detect_markers_params}
    {detect_markers_returns}
    """
    # Normalize marker family input to a list and create detectors for each
    families: list[str] = (
        marker_family if isinstance(marker_family, list) else [marker_family]
    )

    # Use provided detector_parameters or create a default instance
    if detector_parameters is None:
        detector_parameters = cv2.aruco.DetectorParameters()

    detectors: list[tuple[str, str, cv2.aruco.ArucoDetector]] = []
    for fam in families:
        fam_type, aruco_dict = marker_family_to_dict(fam)
        detectors.append(
            (fam, fam_type, cv2.aruco.ArucoDetector(aruco_dict, detector_parameters))
        )

    if step < 1:
        raise ValueError("step must be >= 1")

    # Resolve preprocessing configuration
    preprocess_kwargs: dict | None = None
    if preprocess is not None:
        preset_name = preprocess
        if preset_name not in PREPROCESS_PRESETS:
            raise ValueError(
                f"Unknown preprocess preset '{preset_name}'. "
                f"Available presets: {list(PREPROCESS_PRESETS)}. "
                "Pass preprocess=None to disable preprocessing."
            )
        preprocess_kwargs = dict(PREPROCESS_PRESETS[preset_name])
        if preprocess_params:
            preprocess_kwargs.update(preprocess_params)
    elif preprocess_params:
        preprocess_kwargs = dict(preprocess_params)

    start_frame_idx, end_frame_idx = resolve_processing_window(
        video,
        processing_window,
        processing_window_unit,
    )

    def _process_frame(frame_idx: int, gray_frame: np.ndarray) -> list[dict]:
        """Run detection on a single grayscale frame across all detectors."""
        records: list[dict] = []
        for fam_name, fam_type, det in detectors:
            all_corners, all_ids, _ = det.detectMarkers(gray_frame)
            if all_ids is None:
                continue

            for corners, marker_id in zip(all_corners, all_ids):
                corners = corners.reshape((4, 2))
                if fam_type == "april":
                    # For AprilTags, corners start with bottom right
                    # For ArUco, corners start with top left
                    # See https://stackoverflow.com/questions/79044142
                    corners = corners[[2, 3, 0, 1], :]
                center = np.mean(corners, axis=0)
                if undistort:
                    corners = distort_points(video, corners)
                    center = distort_points(video, center)
                records.append(
                    {
                        "timestamp [ns]": video.ts[frame_idx],
                        "frame index": frame_idx,
                        "marker family": fam_name,
                        "marker id": int(marker_id[0]),
                        "marker name": f"{fam_name}_{marker_id[0]}",
                        "top left x [px]": corners[0, 0],
                        "top left y [px]": corners[0, 1],
                        "top right x [px]": corners[1, 0],
                        "top right y [px]": corners[1, 1],
                        "bottom right x [px]": corners[2, 0],
                        "bottom right y [px]": corners[2, 1],
                        "bottom left x [px]": corners[3, 0],
                        "bottom left y [px]": corners[3, 1],
                        "center x [px]": center[0],
                        "center y [px]": center[1],
                    }
                )
        return records

    detected_markers = []
    frames_to_process = list(range(start_frame_idx, end_frame_idx + 1, step))

    # Ensure video is at the beginning before processing
    video.reset()

    for frame_index in tqdm(frames_to_process, desc="Detecting markers"):
        frame = video.read_frame_at(frame_index)
        if frame is None:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if undistort:
            gray_frame = video.undistort_frame(gray_frame)
        if preprocess_kwargs is not None:
            gray_frame = preprocess_marker_frame(gray_frame, **preprocess_kwargs)
        records = _process_frame(frame_index, gray_frame)
        detected_markers.extend(records)

    if not detected_markers:
        raise ValueError(
            f"No {families} marker detected with the specified parameters."
        )

    df = pd.DataFrame(detected_markers)
    df.set_index("timestamp [ns]", inplace=True)
    _validate_df_columns(df, MARKER_DETECTION_COLUMNS, df_name="marker detections")
    return Stream(df)
