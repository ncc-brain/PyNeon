class SafeDict(dict):
    """A dict that returns the placeholder itself if the key is missing."""

    def __missing__(self, key):
        return "{" + key + "}"


def interp_doc(func):
    """
    Decorator to inject common interpolation parameter docs into a function docstring.
    Leaves other placeholders intact if they are not provided.
    """
    text = (
        "float_kind : str or int, optional\n"
        "   Kind of interpolation applied to columns of float type.\n"
        "   See :class:`scipy.interpolate.interp1d` for details.\n"
        '   Defaults to "linear".\n'
        "other_kind : str or int, optional\n"
        "   Kind of interpolation applied to columns of other types.\n"
        "   See :class:`scipy.interpolate.interp1d` for details.\n"
        '   Only "nearest", "nearest-up", "previous", and "next" are recommended.\n'
        '   Defaults to "nearest".\n'
    )
    if func.__doc__:  # only do formatting if docstring exists
        func.__doc__ = func.__doc__.format_map(SafeDict(interp_doc=text))
    return func


def inplace_doc(func):
    """
    Decorator to inject the common 'inplace' parameter docs into a function docstring.
    Leaves other placeholders intact if they are not provided.
    """
    text = (
        "inplace : bool, optional\n"
        "   If ``True``, replace current data. Otherwise returns a new instance.\n"
        "   Defaults to ``False``.\n"
    )
    if func.__doc__:
        func.__doc__ = func.__doc__.format_map(SafeDict(inplace_doc=text))
    return func
