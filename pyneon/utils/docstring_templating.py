"""Docstring templating and rendering engine for PyNeon.

This module provides:
- `render_doc()`: Low-level utility for rendering docstring templates with
  `{}`-style placeholders and nested template resolution.
- `fill_doc`: Decorator for applying PyNeon's `DOC` registry to function docstrings.
- `DOC`: PyNeon's central registry of reusable docstring snippets.

The system supports nested templates (e.g., `{detect_markers_params}` containing
`{step_param}` and `{window_params}`) with multi-pass recursive resolution.
"""

from __future__ import annotations

from typing import Mapping


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        # Leave unknown placeholders intact to avoid KeyError during
        # incremental migration.
        return "{" + key + "}"


def render_doc(
    template: str,
    *,
    mapping: Mapping[str, str] | None = None,
    max_passes: int = 10,
) -> str:
    """Render a documentation template.

    Parameters
    ----------
    template : str
        Docstring template using ``{key}`` placeholders.
    mapping : Mapping[str, str], optional
        Mapping of keys available to format into the template.
    max_passes : int, optional
        Maximum number of recursive formatting passes used to resolve nested
        placeholders (e.g., ``{outer}`` -> ``{inner}`` -> value).
        Defaults to 10.

    Returns
    -------
    str
        The rendered docstring.
    """
    if not template:
        return template

    rendered = template

    fmt_map = _SafeDict()
    if mapping:
        fmt_map.update(mapping)

    # Resolve placeholders with multiple passes so nested templates in mapping
    # entries are expanded (e.g., {detect_markers_params} containing {step_param}).
    seen: set[str] = {rendered}
    for _ in range(max_passes):
        try:
            next_rendered = rendered.format_map(fmt_map)
        except Exception:
            # If formatting fails for any reason, return the most recent
            # successfully rendered string so docs remain available.
            return rendered

        if next_rendered == rendered:
            break

        rendered = next_rendered

        # Stop if a previous intermediate state repeats (cyclic template refs).
        if rendered in seen:
            break
        seen.add(rendered)

    return rendered


# PyNeon's central registry of reusable docstring snippets
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


DOC["inplace_param"] = """\
inplace : bool, optional
    If ``True``, replace current data. Otherwise returns a new instance.
    Defaults to ``False``."""

DOC["step_param"] = """\
step : int, optional
    Process every Nth frame. For example, step=5 processes frames
    0, 5, 10, 15, .... Defaults to 1 (process all frames)."""

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

DOC["max_gap_ms_param"] = """\
max_gap_ms : int, optional
    Maximum allowed distance (in milliseconds) to both adjacent original
    timestamps (left and right). A requested new timestamp will be ignored
    if its distance to the immediate left OR right original timestamp is
    greater than or equal to ``max_gap_ms`` (no interpolation will be
    performed at that timestamp). Defaults to 500."""

DOC["stream_or_none_returns"] = """\
Stream or None
    A new :class:`Stream` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["events_or_none_returns"] = """\
Events or None
    A new :class:`Events` instance with modified data
    if ``inplace=False``, otherwise ``None``.
"""

DOC["epochs_info"] = """\
epochs_info : pandas.DataFrame, shape (n_epochs, 4)
    DataFrame containing epoch information with the following columns
    (times are UNIX timestamps in nanoseconds):

    ============ ================================
    Column       Description
    ============ ================================
    t_ref        Reference time of the epoch.
    t_before     Time before the reference time to start the epoch.
    t_after      Time after the reference time to end the epoch.
    description  Description or label associated with the epoch.
    ============ ================================
"""

DOC["detect_markers_params"] = """
marker_family : str or list[str], optional
    AprilTag family/ArUco dictionary to detect. Accepts a single family string
    (e.g., '36h11') or a list of families (e.g., ['36h11', '6x6_250']).
{step_param}
{window_params}
detector_parameters : cv2.aruco.DetectorParameters, optional
    Detector parameters to use for all marker families. If None, a default
    DetectorParameters instance is created. Defaults to ``None``.
undistort : bool, optional
    If True, undistorts frames before detection, which can improve detection
    performance, then redistorts detected points. Returned coordinates remain
    in the original (distorted) video frame. Defaults to ``False``.
preprocess : bool or str, optional
    Enable a grayscale preprocessing stage applied after optional undistortion
    and before marker detection. Preprocessing can improve robustness under
    challenging conditions such as low contrast, uneven illumination, or
    IR-emitter hotspots.

    Accepted values:

    - ``False`` (default): no preprocessing; current behaviour is preserved.
    - ``True``: apply the ``"mild"`` preset (CLAHE + highlight clipping +
      light blur + unsharp mask).
    - ``"mild"``: balanced general-purpose preset.
    - ``"ir"``: stronger highlight clipping for IR-emitter hotspots.
    - ``"low_light"``: more aggressive CLAHE for dim scenes.

    Individual preprocessing parameters can be further overridden via
    ``preprocess_params``. Defaults to ``False``.
preprocess_params : dict or None, optional
    Keyword arguments forwarded to
    :func:`~pyneon.video.marker.preprocess_marker_frame` that override the
    values of the active preset (or set values when ``preprocess=False``).
    Allowed keys are: ``clahe``, ``clahe_clip_limit``,
    ``clahe_tile_grid_size``, ``clip_highlights``, ``highlight_percentile``,
    ``gaussian_blur_sigma``, ``sharpen``, ``sharpen_amount``.
    Defaults to ``None``.
"""

DOC["detect_contour_params"] = """
{step_param}
{window_params}
min_area_ratio : float, optional
    Minimum contour area relative to frame area. Contours smaller than this
    ratio are ignored. Default is 0.01 (1 percent of frame area).
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

DOC["marker_layout"] = """
marker_layout : pandas.DataFrame
    DataFrame describing the spatial layout of the markers on the surface.
    Must contain the following columns:

    {marker_layout_table}
"""

DOC["marker_layout_table"] = """
    =========== =========================================================
    Column      Description
    =========== =========================================================
    marker name Full marker name (family_id, e.g., "36h11_1")
    size        Size of the marker in surface units
    center x    x-coordinate of the marker center in surface coordinates
    center y    y-coordinate of the marker center in surface coordinates
    =========== =========================================================
"""

DOC["homographies"] = """
homographies : Stream
    Stream indexed by "timestamp [ns]" with columns "homography (0,0)"
    through "homography (2,2)", corresponding to the 9 elements of the
    estimated 3x3 homography matrix for each retained frame.
"""

DOC["detect_markers_returns"] = """
Stream
    Stream of detected markers. Each row corresponds to a detected marker in a video frame
    indexed by "timestamp [ns]" and contains the following columns:

    =================== =========================================================
    Column              Description
    =================== =========================================================
    frame index         Frame number of the marker detection.
    marker family       AprilTag family or ArUco dictionary
                        of the detected marker (e.g., "36h11", "6x6_250").
    marker id           ID of the detected marker within its family (e.g., 0, 1, 2).
    marker name         Full identifier combining family and id (e.g., "36h11_0", "36h11_1").
    top left x [px]     X coordinate of the top-left corner of the detected marker.
    top left y [px]     Y coordinate of the top-left corner of the detected marker.
    top right x [px]    X coordinate of the top-right corner of the detected marker.
    top right y [px]    Y coordinate of the top-right corner of the detected marker.
    bottom right x [px] X coordinate of the bottom-right corner of the detected marker.
    bottom right y [px] Y coordinate of the bottom-right corner of the detected marker.
    bottom left x [px]  X coordinate of the bottom-left corner of the detected marker.
    bottom left y [px]  Y coordinate of the bottom-left corner of the detected marker.
    center x [px]       X coordinate of the center of the detected marker.
    center y [px]       Y coordinate of the center of the detected marker.
    =================== =========================================================
"""

DOC["detect_contour_returns"] = """
Stream
    Stream of detected contour coordinates. Each row corresponds to a detected contour in a video frame
    indexed by "timestamp [ns]" and contains the following columns:

    =================== =========================================================
    Column              Description
    =================== =========================================================
    frame index         Frame number of the contour detection.
    contour name        Identifier for the detected contour (e.g., "contour_0").
    top left x [px]     X coordinate of the top-left corner of the detected contour.
    top left y [px]     Y coordinate of the top-left corner of the detected contour.
    top right x [px]    X coordinate of the top-right corner of the detected contour.
    top right y [px]    Y coordinate of the top-right corner of the detected contour.
    bottom right x [px] X coordinate of the bottom-right corner of the detected contour.
    bottom right y [px] Y coordinate of the bottom-right corner of the detected contour.
    bottom left x [px]  X coordinate of the bottom-left corner of the detected contour.
    bottom left y [px]  Y coordinate of the bottom-left corner of the detected contour.
    center x [px]       X coordinate of the center of the detected contour.
    center y [px]       Y coordinate of the center of the detected contour.
    area_ratio          Area of the detected contour relative to frame area (if `report_diagnostics` is True).
    score               Diagnostic score indicating how closely the detected contour resembles a perfect rectangle
                        (if `report_diagnostics` is True).
    =================== =========================================================
"""

DOC["fig_ax_returns"] = """
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
    """Apply the project's `DOC` mapping to a function's docstring.

    This decorator resolves brace-style placeholders (e.g., `{key}`) from
    the project's `DOC` registry into the docstring. Supports nested
    placeholders (e.g., `{detect_markers_params}` expands to include
    `{step_param}`, `{window_params}`, etc.).

    Examples
    --------
    Basic usage::

        @fill_doc
        def detect_markers(marker_family="36h11"):
            '''Detect markers.

            Parameters
            ----------
            {detect_markers_params}
            '''

    When applied, `{detect_markers_params}` is replaced with the full
    parameter documentation from `DOC["detect_markers_params"]`.
    """
    if func.__doc__:
        func.__doc__ = render_doc(func.__doc__, mapping=DOC)
    return func


if __name__ == "__main__":
    print(DOC["detect_markers_params"])
