import json
import pytest

@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
@pytest.mark.parametrize(
    "output_dir",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
def test_export_motion_bids(request, dataset_fixture):
    dataset = request.getfixturevalue(dataset_fixture)
    for recording in dataset.recordings:
        recording.export_motion_bids("data/export")