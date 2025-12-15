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


def fill_doc(func):
    """Fill a function docstring with common doc snippets using %-format."""
    if func.__doc__:
        func.__doc__ = func.__doc__ % DOC
    return func
