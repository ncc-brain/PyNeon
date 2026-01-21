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

DOC["detect_apriltags_params"] = """
tag_family : str, optional
    Tag families to detect, separated by space (default 'tag36h11').
nthreads : int, optional
    Number of threads for detection (default 4).
quad_decimate : float, optional
    Detection of quads can be done on a lower-resolution image, improving speed
    at a cost of pose accuracy and a slight decrease in detection rate. Decoding
    the binary payload is still done at full resolution (default 1.0).
quad_sigma : float, optional
    Gaussian blur applied to the segmented image for quad detection. Standard
    deviation in pixels. Noisy images benefit from non-zero values (e.g. 0.8),
    default: 0.0
refine_edges : int, optional
    When non-zero, the edges of each quad are adjusted to "snap to" strong
    gradients nearby. Useful when decimation is employed and increases the quality
    of the initial quad estimate. Computationally inexpensive (default 1).
decode_sharpening : float, optional
    Amount of sharpening applied to decoded images. Can help decode small tags but
    may affect performance in odd lighting or low light conditions (default 0.25).
debug : int, optional
    If 1, will save debug images (runs very slow, default 0).
step : int, optional
    If > 1, detect tags only in every Nth frame.
    E.g., step=5 will process frames 0, 5, 10, 15, etc.
detection_window : tuple, optional
    A tuple (start, end) specifying the range to search for AprilTag detections.
    The interpretation depends on `detection_window_unit`. Defaults to ``None`` (all frames).
detection_window_unit : {"frame", "time", "timestamp"}, optional
    Unit for the values in `detection_window`:
        - "timestamp": start and end are absolute timestamps in nanoseconds
        - "time": start and end are in seconds (relative to video start)
        - "frame": start and end are frame indices (0-based)
    Defaults to "frame".
**detector_kwargs : optional
    Additional keyword arguments passed to the AprilTag detector for fine-tuning.
"""

DOC["detect_apriltags_return"] = """
Stream
    Stream indexed by 'timestamp [ns]' with columns:
    - 'frame id': The frame number
    - 'tag id': The ID of the detected AprilTag
    - 'corner 0 x [px]', 'corner 0 y [px]': First corner (TL)
    - 'corner 1 x [px]', 'corner 1 y [px]': Second corner (TR)
    - 'corner 2 x [px]', 'corner 2 y [px]': Third corner (BR)
    - 'corner 3 x [px]', 'corner 3 y [px]': Fourth corner (BL)
    - 'center x [px]': X-coordinate of the tag center in pixels
    - 'center y [px]': Y-coordinate of the tag center in pixels
"""


def fill_doc(func):
    """Fill a function docstring with common doc snippets using %-format."""
    if func.__doc__:
        func.__doc__ = func.__doc__ % DOC
    return func
