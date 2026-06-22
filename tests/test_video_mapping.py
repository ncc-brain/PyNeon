import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pyneon import Video, find_homographies, plot_marker_layout

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Failed to retrieve frame at index \d+\. Returning None\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Recording .* misses the following expected files:UserWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*requested timestamps are outside the data time range and will have empty data\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*requested timestamps exceed max_gap_ms=\d+ relative to neighboring samples and will have empty data\.:UserWarning"
    ),
]

width = 2560
height = 1440
apriltag_layout = pd.DataFrame(
    {
        "marker name": [f"36h11_{i}" for i in range(6)],
        "size": 200,
        "center x": [150, width - 150, width - 150, width - 150, 150, 150],
        "center y": [150, 150, height / 2, height - 150, height - 150, height / 2],
    }
)
aruco_layout = pd.DataFrame(
    {
        "marker name": [f"5x5_50_{i}" for i in range(4)],
        "size": 200,
        "center x": [150, width - 150, width - 150, 150],
        "center y": [150, 150, height - 150, height - 150],
    }
)
contour_layout = np.array(
    [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ],
)


@pytest.mark.parametrize(
    "ax",
    [None, plt.subplots()[1]],
)
@pytest.mark.parametrize(
    "layout",
    [apriltag_layout, aruco_layout],
)
def test_plot_marker_layout(ax, layout):
    plot_marker_layout(layout, ax=ax, show=False)
    plt.close("all")


@pytest.mark.parametrize(
    "dataset_fixture",
    ["mapping_dataset_native", "mapping_dataset_cloud"],
)
def test_mapping(request, dataset_fixture, tmp_path):
    dataset = request.getfixturevalue(dataset_fixture)
    for recording in dataset.recordings:
        video: Video = recording.scene_video

        if "april" in recording.recording_dir.name:
            layout = apriltag_layout
            marker_detections = video.detect_markers("36h11", step=2, undistort=True)
        elif "aruco" in recording.recording_dir.name:
            layout = aruco_layout
            marker_detections = video.detect_markers("5x5_50", step=2, undistort=True)
        else:
            raise ValueError("Unknown recording, cannot determine marker type")

        contour_detections = video.detect_contour(step=2, undistort=True)

        # Pick a random frame from detections to plot
        random_marker_ts = int(marker_detections.data.sample(n=1).index[0])
        random_marker_index = int(
            video.timestamp_to_frame_index(random_marker_ts).item()
        )
        random_contour_ts = int(contour_detections.data.sample(n=1).index[0])
        random_contour_index = int(
            video.timestamp_to_frame_index(random_contour_ts).item()
        )
        fig, axs = plt.subplots(1, 2)
        video.plot_detections(
            marker_detections, frame_index=random_marker_index, ax=axs[0], show=False
        )
        video.plot_detections(
            contour_detections, frame_index=random_contour_index, ax=axs[1], show=False
        )
        plt.close(fig)

        # Test that we can save videos with detections overlaid
        video.overlay_detections(
            marker_detections, output_path=tmp_path / "detections.mp4"
        )
        video.overlay_detections(
            contour_detections, output_path=tmp_path / "contours.mp4"
        )

        # Estimate homographies and apply to gaze and fixations
        marker_homographies = find_homographies(
            marker_detections,
            layout,
        )
        contour_homographies = find_homographies(
            contour_detections,
            contour_layout,
        )

        gaze = recording.gaze
        for homographies in [marker_homographies, contour_homographies]:
            gaze_on_surface = gaze.apply_homographies(homographies)
            assert "gaze x [surface coord]" in gaze_on_surface.columns
            assert "gaze y [surface coord]" in gaze_on_surface.columns

        if dataset_fixture == "mapping_dataset_cloud":
            fixations = recording.fixations
            for homographies in [marker_homographies, contour_homographies]:
                fixations_on_surface = fixations.apply_homographies(homographies)
                assert "fixation x [surface coord]" in fixations_on_surface.columns
                assert "fixation y [surface coord]" in fixations_on_surface.columns
