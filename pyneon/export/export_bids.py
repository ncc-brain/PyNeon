from pathlib import Path
import pandas as pd
import json
import datetime
import re
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
    """
    Export IMU data to Motion-BIDS format. Continuous samples are saved to a .tsv
    file and metadata (with template fields) are saved to a .json file.
    Users should later edit the metadata file according to the experiment to make
    it BIDS-compliant.

    Parameters
    ----------
    rec : :class:`NeonRecording`
        Recording object containing the IMU data.
    motion_dir : str or :class:`pathlib.Path`
        Output directory to save the Motion-BIDS formatted data.
    prefix : str, optional
        Prefix for the BIDS filenames, by default "sub-XX_task-YY_tracksys-NeonIMU".
        The format should be `sub-<label>[_ses-<label>]_task-<label>_tracksys-<label>[_acq-<label>][_run-<index>]`
        (Fields in [] are optional). Files will be saved as
        ``{prefix}_motion.<tsv|json>`` and ``{prefix}_channels.tsv``.
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
        raise FileNotFoundError(f"Directory not found: {motion_dir}")
    if motion_dir.name != "motion":
        raise RuntimeWarning(
            f"Directory name {motion_dir.name} is not 'motion' as specified by Motion-BIDS"
        )
    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    channels_tsv_path = motion_dir / f"{prefix}_channels.tsv"

    imu = rec.imu
    if imu is None:
        raise ValueError("No IMU data found in the recording.")
    interp_data = imu.interpolate()
    motion_first_ts = interp_data.loc[0, "timestamp [ns]"]
    motion_acq_time = datetime.datetime.fromtimestamp(motion_first_ts / 1e9).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    interp_data = interp_data.drop(columns=["timestamp [ns]", "time [s]"])

    interp_data.to_csv(
        motion_tsv_path, sep="\t", index=False, header=False, na_rep="n/a"
    )

    ch_names = interp_data.columns
    ch_names = [re.sub(r"\s\[[^\]]*\]", "", ch) for ch in ch_names]
    channels = pd.DataFrame(
        {
            "name": ch_names,
            "component": ["x", "y", "z"] * 3 + ["w", "x", "y", "z"],
            "type": ["GYRO"] * 3 + ["ACCEL"] * 3 + ["ORNT"] * 7,
            "placement": ["head-mounted frame"] * 13,
            "units": ["deg/s"] * 3 + ["g"] * 3 + ["deg"] * 3 + ["arbitrary"] * 4,
        }
    )
    channels.to_csv(channels_tsv_path, sep="\t", index=False)

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
        match = re.search(r"(sub-\d+)(_ses-\d+)?", prefix)
        if match:
            scan_prefix = match.group(0)
        else:
            scan_prefix = "sub-XX_ses-YY"
        scans_path = scans_dir / f"{scan_prefix}_scans.tsv"
        scans = new_scan
    scans.to_csv(scans_path, sep="\t", index=False)


def export_eye_bids(rec: "NeonRecording", output_dir: Union[str, Path]):
    gaze = rec.gaze
    eye_states = rec.eye_states
    output_dir = Path(output_dir)
    pass
