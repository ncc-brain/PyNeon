import pytest
import pyneon
from pyneon import (
    Dataset,
    Recording,
    Stream,
    Events,
    Video,
    Epochs,
    get_sample_data,
)


def test_basic_import():
    assert hasattr(pyneon, "__version__")


@pytest.mark.parametrize("dataset_name", ["boardView", "screenFlash"])
def test_get_sample_data(dataset_name):
    sample_dir = get_sample_data(dataset_name)
    dataset = Dataset(sample_dir / "Timeseries Data + Scene Video")
    for recording in dataset.recordings:
        assert isinstance(recording, Recording)
