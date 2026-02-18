DOC = dict()

DOC["interp_kind_params"] = """\
float_kind : str or int, optional
    Kind of interpolation applied to columns of float type.
    See :class:`scipy.interpolate.interp1d` for details.
    Defaults to "linear".
other_kind : str or int, optional
    Kind of interpolation applied to columns of other types.
    See :class:`scipy.interpolate.interp1d` for details.
    Only "nearest", "nearest-up", "previous", and "next" are recommended.
    Defaults to "nearest".
"""


DOC["inplace"] = """\
inplace : bool, optional
    If ``True``, replace current data. Otherwise returns a new instance.
    Defaults to ``False``."""

DOC["step_param"] = """\
step : int, optional
    Process every Nth frame. For example, step=5 processes frames
    0, 5, 10, 15, ...). Defaults to 1 (process all frames)."""

DOC["window_params"] = """\
processing_window : tuple[int | float, int | float] or None
    Start and end of the processing window. Interpretation depends on
    ``processing_window_unit``. Defaults to ``None`` (full duration).
processing_window_unit : {"frame", "time", "timestamp"}, optional
    Unit for values in ``processing_window``. Possible values are:

    - "timestamp": Unix timestamps in nanoseconds
    - "time": Seconds relative to video start
    - "frame": video frame indices (0-based)

    Defaults to "frame".
"""

DOC["max_gap_ms"] = """\
max_gap_ms : int, optional
    Maximum allowed distance (in milliseconds) to both adjacent original
    timestamps (left and right). A requested new timestamp will be ignored
    if its distance to the immediate left OR right original timestamp is
    greater than or equal to ``max_gap_ms`` (no interpolation will be
    performed at that timestamp). Defaults to 500."""

DOC["stream_or_none"] = """\
Stream or None
    A new :class:`Stream` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["events_or_none"] = """\
Events or None
    A new :class:`Events` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["epochs_info"] = """\
epochs_info : pandas.DataFrame, shape (n_epochs, 4)
    DataFrame containing epoch information with the following columns
    (times are UNIX timestamps in nanoseconds):

    ================ ================================
    Column           Description
    ================ ================================
    ``t_ref``        Reference time of the epoch.
    ``t_before``     Time before the reference time to start the epoch.
    ``t_after``      Time after the reference time to end the epoch.
    ``description``  Description or label associated with the epoch.
    ================ ================================
"""

DOC["detect_markers_params"] = """
marker_family : str or list[str], optional
    AprilTag family/ArUco dictionary to detect. Accepts a single family string
    (e.g., '36h11') or a list of families (e.g., ['36h11', '6x6_250']).
%(step_param)s
%(window_params)s
detector_parameters : cv2.aruco.DetectorParameters, optional
    Detector parameters to use for all marker families. If None, a default
    DetectorParameters instance is created. Defaults to ``None``.
undistort : bool, optional
    If True, undistorts frames before detection, which can improve detection
    performance, then redistorts detected points. Returned coordinates remain
    in the original (distorted) video frame. Defaults to ``False``.
"""

DOC["detect_surface_params"] = """
%(step_param)s
%(window_params)s
min_area_ratio : float, optional
    Minimum contour area relative to frame area. Contours smaller than this
    ratio are ignored. Default is 0.01 (1% of frame area).
max_area_ratio : float, optional
    Maximum contour area relative to frame area. Contours larger than this
    ratio are ignored. Default is 0.98.
brightness_threshold : int, optional
    Fixed threshold for binarization when `adaptive=False`. Default is 180.
adaptive : bool, optional
    If True (default), use adaptive thresholding to handle varying
    illumination across frames.
morph_kernel : int, optional
    Kernel size for morphological closing (default 5). Use 0 to disable
    morphological operations.
decimate : float, optional
    Downsampling factor for faster processing (e.g., 0.5 halves resolution).
    Detected coordinates are automatically rescaled back. Default is 1.0.
mode : {"largest", "best", "all"}, optional
    Selection mode determining which contours to return per frame:

    - "largest" : Return only the largest valid rectangular contour.
      Useful when the surface is the outermost bright region. (Default)
    - "best" : Return the contour that most closely resembles a
      perfect rectangle (lowest corner-angle variance and balanced
      aspect ratio).
    - "all" : Return all valid rectangular contours (outer and inner
      overlapping rectangles). Useful when both surface and inner
      projected content need to be distinguished.
report_diagnostics : bool, optional
    If True, includes "area_ratio" and "score" columns in the output.
    Defaults to False.
undistort : bool, optional
    If True, undistorts frames before detection, which can improve detection
    performance, then redistorts detected points. Returned coordinates remain
    in the original (distorted) video frame. Defaults to ``False``.
"""

DOC["find_homographies_params"] = """
valid_markers : int, optional
    Minimum number of markers required to compute a homography. Defaults to 2.
method : int, optional
    Method used to compute a homography matrix. The following methods are possible:

    - 0 - a regular method using all the points, i.e., the least squares method
    - ``cv2.RANSAC`` - RANSAC-based robust method
    - ``cv2.LMEDS`` - Least-Median robust method
    - ``cv2.RHO`` - PROSAC-based robust method

    Defaults to ``cv2.LMEDS``.
ransacReprojThreshold : float, optional
    Maximum allowed reprojection error to treat a point pair as an inlier
    (used in the RANSAC and RHO methods only). Defaults to 3.0.
maxIters : int, optional
    The maximum number of RANSAC iterations. Defaults to 2000.
confidence : float, optional
    Confidence level, between 0 and 1, for the estimated homography.
    Defaults to 0.995.
"""

DOC["marker_layout"] = """
marker_layout : pandas.DataFrame
    DataFrame describing the layout. Must include columns:

    - "marker name": full marker identifier (family + id, e.g., "tag36h11_1")\n
    - "size": marker size in the reference plane units\n
    - "center x": x center of the marker in surface coordinates\n
    - "center y": y center of the marker in surface coordinates
"""

DOC["find_homographies_return"] = """
Stream
    A Stream indexed by 'timestamp [ns]' with columns
    'homography (0,0)' through 'homography (2,2)': The 9 elements of the
    flattened 3x3 homography matrix.
"""

DOC["detect_markers_return"] = """
Stream
    Stream indexed by "timestamp [ns]" with columns:

    - "frame index": The frame number\n
    - "marker family": Marker family (e.g., "36h11")\n
    - "marker id": Marker ID, for example 0, 1\n
    - "marker name": Marker identifier, for example "36h11_0", "36h11_1"\n
    - "top left x [px]", "top left y [px]"\n
    - "top right x [px]", "top right y [px]"\n
    - "bottom right x [px]", "bottom right y [px]"\n
    - "bottom left x [px]", "bottom left y [px]"\n
    - "center x [px]": X-coordinate of marker center in pixels\n
    - "center y [px]": Y-coordinate of marker center in pixels
"""

DOC["detect_surface_return"] = """
Stream
    Stream indexed by "timestamp [ns]" with columns:

    - "frame index": The frame number\n
    - "marker family": "surface"\n
    - "marker id": sequential ID per contour in frame\n
    - "marker name": identifier like "surface_0"\n
    - "top left x [px]", "top left y [px]"\n
    - "top right x [px]", "top right y [px]"\n
    - "bottom right x [px]", "bottom right y [px]"\n
    - "bottom left x [px]", "bottom left y [px]"\n
    - "center x [px]": X-coordinate of center in pixels\n
    - "center y [px]": Y-coordinate of center in pixels\n
    - "area_ratio": float (if `report_diagnostics` is True)\n
    - "score": float (if `report_diagnostics` is True)
"""

DOC["fig_ax_return"] = """
fig : matplotlib.figure.Figure
    Figure instance containing the plot.
ax : matplotlib.axes.Axes
    Axis instance containing the plot.
"""

DOC["ax_param"] = """
ax : matplotlib.axes.Axes or None
    Axis to plot on. If ``None``, a new figure is created. Defaults to ``None``.
"""

DOC["show_param"] = """
show : bool
    Show the figure if ``True``. Defaults to True.
"""

DOC["show_video_param"] = """
show_video : bool, optional
    Whether to display the video with overlays in real-time.
    Press 'q' to quit early. Defaults to ``False``.
"""

DOC["output_path_param"] = """
output_path : pathlib.Path or str or None, optional
    Path to save the output video with overlays. If ``None``, the video is not saved.
    Either this or ``show_video=True`` must be provided.
"""


def fill_doc(func):
    """Fill a function docstring with common doc snippets using %-format."""
    if func.__doc__:
        func.__doc__ = func.__doc__ % DOC
    return func
