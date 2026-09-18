import numpy as np
import pytest

from pyneon.video import marker


class _DummyVideo:
    def __init__(self):
        self.ts = np.array([123], dtype=np.int64)

    def reset(self):
        return None

    def read_frame_at(self, frame_index: int):
        return np.zeros((2, 2, 3), dtype=np.uint8)


@pytest.mark.parametrize("raw_marker_id", [7, np.array(7), np.array([7])])
def test_normalize_marker_id_shapes(raw_marker_id):
    normalized_marker_id = marker._normalize_marker_id(raw_marker_id)
    assert isinstance(normalized_marker_id, int)
    assert normalized_marker_id == 7


@pytest.mark.parametrize("raw_marker_id", [7, np.array(7), np.array([7])])
def test_detect_markers_normalizes_marker_id_shapes(monkeypatch, raw_marker_id):
    class _DummyDetector:
        def detectMarkers(self, gray_frame):
            corners = [
                np.array(
                    [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
                    dtype=np.float32,
                )
            ]
            return corners, [raw_marker_id], None

    monkeypatch.setattr(marker, "marker_family_to_dict", lambda _: ("aruco", object()))
    monkeypatch.setattr(marker.cv2.aruco, "ArucoDetector", lambda *_: _DummyDetector())

    detections = marker.detect_markers(_DummyVideo(), marker_family="5x5_50")

    assert detections.data.iloc[0]["marker id"] == "7"
    assert detections.data.iloc[0]["marker name"] == "5x5_50_7"
