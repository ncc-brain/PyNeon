import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pyneon import find_homographies, plot_marker_layout

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
        video = recording.scene_video

        if "april" in recording.recording_dir.name:
            layout = apriltag_layout
            marker_detections = video.detect_markers("36h11", step=2, undistort=True)
        elif "aruco" in recording.recording_dir.name:
            layout = aruco_layout
            marker_detections = video.detect_markers("5x5_50", step=2, undistort=True)
        else:
            raise ValueError("Unknown recording, cannot determine marker type")

        contour_detections = video.detect_contour(step=5, undistort=True)
        video.plot_detections(marker_detections, frame_index=101, show=False)
        video.plot_detections(contour_detections, frame_index=101, show=False)
        video.overlay_detections(
            marker_detections, output_path=tmp_path / "detections.mp4"
        )
        video.overlay_detections(
            contour_detections, output_path=tmp_path / "contours.mp4"
        )
        marker_homographies = find_homographies(
            marker_detections,
            layout,
        )
        contour_homographies = find_homographies(
            contour_detections,
            layout,
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
