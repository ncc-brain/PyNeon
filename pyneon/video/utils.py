import re
from numbers import Number
from typing import Tuple, Iterable
import cv2
import pandas as pd

from .constants import APRILTAG_FAMILIES, ARUCO_NUMBERS, ARUCO_SIZES

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


def _verify_format(df: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    """Verify that the DataFrame contains all expected columns (including index)."""
    actual_columns = set(df.columns)
    if df.index.name:
        actual_columns.add(df.index.name)

    missing = set(expected_columns) - actual_columns
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
