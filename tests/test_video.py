import pytest
import numpy as np
import cv2

@pytest.mark.parametrize(
    "dataset_fixture",
    ["simple_dataset_native", "simple_dataset_cloud"],
)
def test_video_basics(request, dataset_fixture):
    dataset = request.getfixturevalue(dataset_fixture)
    for recording in dataset.recordings:
        if dataset_fixture == "simple_dataset_cloud":
            with pytest.raises(ValueError, match="Pupil Cloud recordings do not contain eye video."):
                eye_video = recording.eye_video
        else:
            eye_video = recording.eye_video
            assert eye_video.info == {}

        video = recording.scene_video
        n_frames = len(video.ts)
        assert n_frames == video.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Select 5 random frames within n_frames
        random_frames = np.random.choice(n_frames, size=1, replace=False)
        for frame_idx in random_frames:
            frame_idx = int(frame_idx)
            frame = video.read_frame_at(frame_idx)
            assert frame.shape == (video.height, video.width, 3)
            assert frame_idx == video.current_frame_index