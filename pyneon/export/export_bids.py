import datetime
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ._bids_parameters import MOTION_META_DEFAULT

if TYPE_CHECKING:
    from ..recording import Recording


def export_motion_bids(
    rec: "Recording",
    motion_dir: str | Path,
    prefix: Optional[str] = None,
    extra_metadata: dict = {},
):
    """
    Export IMU data to Motion-BIDS format. Continuous samples are saved to a .tsv
    file and metadata (with template fields) are saved to a .json file.
    Users should later edit the metadata file according to the experiment to make
    it BIDS-compliant.

    Parameters
    ----------
    rec : Recording
        Recording instance containing the IMU data.
    motion_dir : str or pathlib.Path
        Output directory to save the Motion-BIDS formatted data.
    prefix : str, optional
        Prefix for the BIDS filenames, by default "sub-``wearer_name``_task-XXX_tracksys-NeonIMU".
        The format should be "sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]"
        (Fields in [] are optional). Files will be saved as
        ``{prefix}_motion.<tsv|json>``.
    extra_metadata : dict, optional
        Additional metadata to include in the JSON file. Defaults to an empty dict.

    Notes
    -----
    Motion-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
    standardize the organization of motion data for reproducible research [1]_.
    For more information, see
    https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html.

    References
    ----------
    .. [1] Jeung, S., Cockx, H., Appelhoff, S., Berg, T., Gramann, K., Grothkopp, S., ... & Welzel, J. (2024). Motion-BIDS: an extension to the brain imaging data structure to organize motion data for reproducible research. *Scientific Data*, 11(1), 716.
    """

    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        motion_dir.mkdir(parents=True)
    if motion_dir.name != "motion":
        raise RuntimeWarning(
            f"Directory name {motion_dir.name} is not 'motion' as specified by Motion-BIDS"
        )
    if prefix is None:
        prefix = f"sub-{rec.info['wearer_name']}_task-XXX_tracksys-NeonIMU"

    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    channels_tsv_path = motion_dir / f"{prefix}_channels.tsv"
    channels_json_path = motion_dir / f"{prefix}_channels.json"

    imu = rec.imu
    if imu is None:
        raise ValueError("No IMU data found in the recording.")
    imu = imu.interpolate()
    motion_acq_time = datetime.datetime.fromtimestamp(imu.first_ts / 1e9).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )

    imu.data.to_csv(motion_tsv_path, sep="\t", index=False, header=False, na_rep="n/a")

    ch_names = imu.columns
    ch_names = [re.sub(r"\s\[[^\]]*\]", "", ch) for ch in ch_names]
    channels = pd.DataFrame(
        {
            "name": ch_names,
            "component": ["x", "y", "z"] * 3 + ["w", "x", "y", "z"],
            "type": ["GYRO"] * 3 + ["ACCEL"] * 3 + ["ORNT"] * 7,
            "tracked_point": ["Head"] * 13,
            "units": ["deg/s"] * 3 + ["g"] * 3 + ["deg"] * 3 + ["arbitrary"] * 4,
            "sampling_frequency": [int(imu.sampling_freq_effective)] * 13,
        }
    )
    channels.to_csv(channels_tsv_path, sep="\t", index=False)

    ch_meta = {
        "reference_frame": {
            "Levels": {
                "global": {
                    "SpatialAxes": "RAS",
                    "RotationOrder": "ZXY",
                    "RotationRule": "right-hand",
                    "Description": "This global reference frame is defined by the IMU axes: X right, Y anterior, Z superior. The scene camera frame differs from this frame by a 102-degree rotation around the X-axis. All motion data are expressed relative to the IMU frame for consistency.",
                },
            }
        }
    }
    with open(channels_json_path, "w") as f:
        json.dump(ch_meta, f, indent=4)

    info = rec.info
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
    filename = [motion_dir.name + "/" + motion_tsv_path.name]
    new_scan = pd.DataFrame.from_dict(
        {
            "filename": filename,
            "acq_time": [motion_acq_time],
        }
    )
    if len(scans_path_potential) >= 1:
        scans_path = scans_path_potential[0]
        scans = pd.read_csv(scans_path, sep="\t")
        if filename in scans.filename.values:
            return
        else:
            scans = pd.concat([scans, new_scan], ignore_index=True)
    else:
        match = re.search(r"(sub-\d+)(_ses-\d+)?", prefix)
        if match:
            scan_prefix = match.group(0)
        else:
            scan_prefix = "sub-XX_ses-YY"
        scans_path = scans_dir / f"{scan_prefix}_scans.tsv"
        scans = new_scan
    scans.to_csv(scans_path, sep="\t", index=False)


def export_eye_bids(rec: "Recording", output_dir: str | Path):
    """
    Under development. Export eye tracking data to Eye-BIDS format.
    """
    gaze = rec.gaze
    eye_states = rec.eye_states
    output_dir = Path(output_dir)
    pass
