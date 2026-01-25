import pandas as pd

from pyneon import Recording, get_sample_data
from pyneon.video import find_homographies

# Load a sample recording
rec_dir = (
    get_sample_data("Artworks") / "Timeseries Data + Scene Video" / "artworks-9a141750"
)
rec = Recording(rec_dir)
print(rec)

video = rec.scene_video
detected_markers = video.detect_markers(
    ["36h11", "6x6_250"],
    detection_window=(180, 210), detection_window_unit="frame"
)

# video.plot_detected_markers(detected_markers, frame_index=210)

marker_layout = pd.DataFrame(
    {
        "marker name": [f"36h11_{i}" for i in range(6)],
        "size": 200,
        "center x": [100, 100, 100, 1820, 1820, 1820],
        "center y": [100, 540, 980, 100, 540, 980],
    }
)

homographies = find_homographies(
    detected_markers,
    marker_layout,
)
print(homographies.data.head())