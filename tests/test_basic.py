# tests/test_import.py

def test_basic_import():
    import pyneon
    assert hasattr(pyneon, "__version__")
    from pyneon import Dataset, Recording, Stream, Events, SceneVideo, Epochs
    from pyneon import get_sample_data