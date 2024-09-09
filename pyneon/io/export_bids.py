from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..recording import NeonRecording


def export_motion_bids(
    rec: "NeonRecording",
    motion_dir: Union[str, Path],
    prefix: str = "",
    institute_metadata: dict = None,
):
    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {motion_dir}")
    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    scans_dir = motion_dir.parent
    # Check if sub-<label>[_ses-<label>]_scans.tsv already exists
    scans_path_potential = list(scans_dir.glob("*scans.tsv"))
    if len(scans_path_potential) >= 1:
        scans_path = scans_path_potential[0]
    else:
        scans_path = scans_dir / "scans.tsv"

    imu = rec.imu
    if imu is None:
        raise ValueError("No IMU data found in the recording.")
    resamp_data = imu.resample().drop()
    resamp_data.to_csv(
        motion_tsv_path, sep="\t", index=False, header=False, na_rep="n/a"
    )

    pass


def exports_eye_bids(rec: "NeonRecording", output_dir: Union[str, Path]):
    gaze = rec.gaze
    eye_states = rec.eye_states
    output_dir = Path(output_dir)
    pass
