from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..recording import NeonRecording


def export_motion_bids(rec: "NeonRecording", output_dir: Union[str, Path]):
    imu = rec.imu
    output_dir = Path(output_dir)
    pass


def exports_eye_bids(rec: "NeonRecording", output_dir: Union[str, Path]):
    gaze = rec.gaze
    eye_states = rec.eye_states
    output_dir = Path(output_dir)
    pass
