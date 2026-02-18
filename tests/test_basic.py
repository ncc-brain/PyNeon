import pytest

import pyneon


def test_basic_import():
    assert hasattr(pyneon, "__version__")
