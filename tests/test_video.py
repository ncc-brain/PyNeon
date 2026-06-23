import cv2
import numpy as np
import pytest

from pyneon.video.marker import PREPROCESS_PRESETS, preprocess_marker_frame


@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
def test_video_basics(request, dataset_fixture):
    dataset = request.getfixturevalue(dataset_fixture)
    for recording in dataset.recordings:
        eye_video = None
        video = None
        try:
            if dataset_fixture == "simple_dataset_cloud":
                with pytest.raises(
                    ValueError,
                    match=(
                        "Recording.eye_video cannot be read because: "
                        "Pupil Cloud recordings do not contain eye video."
                    ),
                ):
                    eye_video = recording.eye_video
            else:
                eye_video = recording.eye_video
                assert eye_video.info == {}

            video = recording.scene_video
            n_frames = len(video.ts)
            assert n_frames == video.get(cv2.CAP_PROP_FRAME_COUNT)

            # Select random frames within n_frames and always test first and last frame
            random_frames = np.random.choice(n_frames, size=10, replace=False)
            random_frames = np.append(random_frames, [0, n_frames - 1])
            for frame_idx in random_frames:
                frame_idx = int(frame_idx)
                if frame_idx == 0:
                    with pytest.warns(
                        UserWarning, match="Failed to retrieve frame at index 0"
                    ):
                        frame = video.read_frame_at(frame_idx)
                    assert frame is None
                else:
                    frame = video.read_frame_at(frame_idx)
                    assert frame.shape == (video.height, video.width, 3)
                assert frame_idx == video.current_frame_index

            with pytest.raises(ValueError, match="is out of bounds."):
                video.read_frame_at(-5)
                video.read_frame_at(n_frames + 1)
        finally:
            if eye_video is not None:
                eye_video.close()
            if video is not None:
                video.close()


# ---------------------------------------------------------------------------
# Unit tests for preprocess_marker_frame
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_gray():
    """A deterministic synthetic grayscale uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640), dtype=np.uint8)


def test_preprocess_returns_uint8_same_shape(synthetic_gray):
    out = preprocess_marker_frame(synthetic_gray)
    assert out.shape == synthetic_gray.shape
    assert out.dtype == np.uint8


def test_preprocess_float_input_accepted(synthetic_gray):
    img_float = synthetic_gray.astype(np.float32)
    out = preprocess_marker_frame(img_float)
    assert out.dtype == np.uint8
    assert out.shape == synthetic_gray.shape


def test_preprocess_noop_preserves_image(synthetic_gray):
    """With all stages disabled the output should equal the input."""
    out = preprocess_marker_frame(
        synthetic_gray,
        clahe=False,
        clip_highlights=False,
        gaussian_blur_sigma=0,
        sharpen=False,
    )
    np.testing.assert_array_equal(out, synthetic_gray)


@pytest.mark.parametrize("preset_name", list(PREPROCESS_PRESETS))
def test_preprocess_all_presets(synthetic_gray, preset_name):
    params = PREPROCESS_PRESETS[preset_name]
    out = preprocess_marker_frame(synthetic_gray, **params)
    assert out.shape == synthetic_gray.shape
    assert out.dtype == np.uint8


def test_preprocess_custom_params(synthetic_gray):
    out = preprocess_marker_frame(
        synthetic_gray,
        clahe=True,
        clahe_clip_limit=3.0,
        clahe_tile_grid_size=(4, 4),
        clip_highlights=True,
        highlight_percentile=95.0,
        gaussian_blur_sigma=1.5,
        sharpen=True,
        sharpen_amount=0.5,
    )
    assert out.shape == synthetic_gray.shape
    assert out.dtype == np.uint8


def test_detect_markers_invalid_preprocess_preset():
    """detect_markers should raise for an unknown preset name."""
    from unittest.mock import MagicMock

    from pyneon.video.marker import detect_markers

    mock_video = MagicMock()
    with pytest.raises(ValueError, match="Unknown preprocess preset"):
        detect_markers(mock_video, "36h11", preprocess="nonexistent_preset")
