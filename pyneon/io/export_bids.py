from pathlib import Path
import pandas as pd
import json
import datetime
from typing import Union, TYPE_CHECKING

from ._bids_parameters import MOTION_META_DEFAULT

if TYPE_CHECKING:
    from ..recording import NeonRecording


def export_motion_bids(
    rec: "NeonRecording",
    motion_dir: Union[str, Path],
    prefix: str = "",
    extra_metadata: dict = {},
):
    info = rec.info
    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {motion_dir}")
    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    channels_tsv_path = motion_dir / f"{prefix}_channels.tsv"

    imu = rec.imu
    if imu is None:
        raise ValueError("No IMU data found in the recording.")
    resamp_data = imu.resample()
    motion_first_ts = resamp_data.loc[0, "timestamp [ns]"]
    motion_acq_time = datetime.datetime.fromtimestamp(motion_first_ts / 1e9).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    resamp_data = resamp_data.drop(columns=["timestamp [ns]", "time [s]"])

    resamp_data.to_csv(
        motion_tsv_path, sep="\t", index=False, header=False, na_rep="n/a"
    )

    channels = pd.DataFrame(
        {
            "name": resamp_data.columns,
            "component": ["x", "y", "z"] * 3 + ["w", "x", "y", "z"],
            "type": ["GYRO"] * 3 + ["ACCEL"] * 3 + ["ORNT"] * 7,
            "placement": ["glasses frame"] * 13,
            "units": ["deg/s"] * 3 + ["g"] * 3 + ["deg"] * 4 + ["arbitrary"] * 3,
        }
    )
    channels.to_csv(channels_tsv_path, sep="\t", index=False)

    metadata = MOTION_META_DEFAULT
    metadata.update(
        {
            "DeviceSerialNumber": info["module_serial_number"],
            "SoftwareVersions": (
                f"App version: {info['app_version']}; "
                f"Pipeline version: {info['pipeline_version']}"
            ),
            "SamplingFrequency": imu.sampling_freq_nominal,
        }
    )
    metadata.update(extra_metadata)

    with open(motion_json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    scans_dir = motion_dir.parent
    scans_path_potential = list(scans_dir.glob("*_scans.tsv"))
    new_scan = pd.DataFrame.from_dict(
        {
            "filename": [motion_dir.name + "/" + motion_tsv_path.name],
            "acq_time": [motion_acq_time],
        }
    )
    if len(scans_path_potential) >= 1:
        scans_path = scans_path_potential[0]
        scans = pd.read_csv(scans_path, sep="\t")
        scans = pd.concat([scans, new_scan], ignore_index=True)
    else:
        scans_path = scans_dir / "sub-XXX_scans.tsv"
        scans = new_scan
    scans.to_csv(scans_path, sep="\t", index=False)


def exports_eye_bids(rec: "NeonRecording", output_dir: Union[str, Path]):
    gaze = rec.gaze
    eye_states = rec.eye_states
    output_dir = Path(output_dir)
    pass
