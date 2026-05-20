from pyneon.utils.docstring_templating import fill_doc, render_doc


def test_render_doc_mapping():
    doc = "Value is {VAL} and missing stays {MISSING}"
    out = render_doc(doc, mapping={"VAL": "X"})
    assert "X" in out
    assert "{MISSING}" in out


def test_fill_doc_applies_mapping():
    @fill_doc
    def h(inplace=False):
        """Doc {inplace_param}"""

    assert "inplace" in h.__doc__


def test_fill_doc_resolves_nested_mapping_entries():
    @fill_doc
    def h2():
        """{detect_markers_params}"""

    assert "step : int" in h2.__doc__
    assert "processing_window" in h2.__doc__
