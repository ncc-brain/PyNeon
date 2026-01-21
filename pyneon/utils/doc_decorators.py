DOC = dict()

DOC["interp_kwargs"] = """\
float_kind : str or int, optional
    Kind of interpolation applied to columns of float type.
    See :class:`scipy.interpolate.interp1d` for details.
    Defaults to "linear".
other_kind : str or int, optional
    Kind of interpolation applied to columns of other types.
    See :class:`scipy.interpolate.interp1d` for details.
    Only "nearest", "nearest-up", "previous", and "next" are recommended.
    Defaults to "nearest"."""


DOC["inplace"] = """\
inplace : bool, optional
    If ``True``, replace current data. Otherwise returns a new instance.
    Defaults to ``False``."""

DOC["stream_or_none"] = """\
Stream or None
    A new ``Stream`` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["events_or_none"] = """\
Events or None
    A new ``Events`` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["epochs_info"] = """\
epochs_info : pandas.DataFrame, shape (n_epochs, 4)
    DataFrame containing epoch information with the following columns (time in ns):

        ``t_ref``: Reference time of the epoch.\n
        ``t_before``: Time before the reference time to start the epoch.\n
        ``t_after``: Time after the reference time to end the epoch.\n
        ``description``: Description or label associated with the epoch.
"""

DOC["detect_markers_params"] = """
marker_name : str, optional
    Marker dictionary to detect (AprilTag or ArUco). Examples: '16h5', '25h9',
    '36h10', '36h11', '4x4_50', '6x6_250'. Default 'tag36h11'.
step : int, optional
    If > 1, detect markers only in every Nth frame (e.g., step=5 processes frames
    0, 5, 10, 15, ...). Defaults to 1.
detection_window : tuple, optional
    A tuple (start, end) specifying the range to search for detections.
    Interpretation depends on `detection_window_unit`. Defaults to ``None`` (all frames).
detection_window_unit : {"frame", "time", "timestamp"}, optional
    Unit for values in `detection_window`:
        - "timestamp": Unix timestamps in nanoseconds
        - "time": in seconds relative to video start
        - "frame": video frame indices (0-based)
    Defaults to "frame".
"""

DOC["detect_markers_return"] = """
Stream
    Stream indexed by 'timestamp [ns]' with columns:
    - 'frame id': The frame number
    - 'tag id': Marker ID, for example "36h11_0", "36h11_1"
    - 'corner 0 x [px]', 'corner 0 y [px]': First corner (TL)
    - 'corner 1 x [px]', 'corner 1 y [px]': Second corner (TR)
    - 'corner 2 x [px]', 'corner 2 y [px]': Third corner (BR)
    - 'corner 3 x [px]', 'corner 3 y [px]': Fourth corner (BL)
    - 'center x [px]': X-coordinate of marker center in pixels
    - 'center y [px]': Y-coordinate of marker center in pixels
"""


def fill_doc(func):
    """Fill a function docstring with common doc snippets using %-format."""
    if func.__doc__:
        func.__doc__ = func.__doc__ % DOC
    return func
