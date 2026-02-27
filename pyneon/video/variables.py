DETECTION_COLUMNS = [
    "frame index",
    "top left x [px]",
    "top left y [px]",
    "top right x [px]",
    "top right y [px]",
    "bottom right x [px]",
    "bottom right y [px]",
    "bottom left x [px]",
    "bottom left y [px]",
    "center x [px]",
    "center y [px]",
]

# Marker related constants
APRILTAG_FAMILIES = ["16h5", "25h9", "36h10", "36h11"]
ARUCO_SIZES = ["4x4", "5x5", "6x6", "7x7"]
ARUCO_NUMBERS = ["50", "100", "250", "1000"]
MARKER_DETECTION_COLUMNS = DETECTION_COLUMNS + [
    "marker family",
    "marker id",
    "marker name",
]

SURFACE_DETECTION_COLUMNS = DETECTION_COLUMNS + [
    "surface name",
]

MARKERS_LAYOUT_COLUMNS = [
    "marker name",
    "size",
    "center x",
    "center y",
]

HOMOGRAPHIES_COLUMNS = [
    "homography (0,0)",
    "homography (0,1)",
    "homography (0,2)",
    "homography (1,0)",
    "homography (1,1)",
    "homography (1,2)",
    "homography (2,0)",
    "homography (2,1)",
    "homography (2,2)",
]
