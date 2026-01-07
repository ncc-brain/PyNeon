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


def fill_doc(func):
    """Fill a function docstring with common doc snippets using %-format."""
    if func.__doc__:
        func.__doc__ = func.__doc__ % DOC
    return func
