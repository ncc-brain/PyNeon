from numbers import Number
import re
from typing import Tuple

import cv2

# Valid marker configuration constants
APRILTAG_FAMILIES: list[str] = ["16h5", "25h9", "36h10", "36h11"]
ARUCO_SIZES: list[str] = ["4x4", "5x5", "6x6", "7x7"]
ARUCO_NUMBERS: list[str] = ["50", "100", "250", "1000"]


def marker_family_to_dict(marker_family: str) -> Tuple[str, cv2.aruco.Dictionary]:
    # Check for AprilTag markers
    if marker_family in APRILTAG_FAMILIES:
        dict_name: str = f"DICT_APRILTAG_{marker_family.upper()}"
        aruco_dict: object = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        return "april", aruco_dict

    # Check for Aruco markers (format: {size}_{number})
    aruco_pattern: re.Pattern = re.compile(r"^(\d+)x\1_(\d+)$")
    pattern_match: re.Match | None = aruco_pattern.match(marker_family)

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
