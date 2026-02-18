import datetime
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from warnings import warn

import pandas as pd

from ._bids_parameters import (
    EYE_EVENTS_META_DEFAULT,
    EYE_META_DEFAULT,
    MOTION_META_DEFAULT,
)

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
    # First check if IMU data is present in the recording
    try:
        imu = rec.imu
    except Exception as e:
        raise ValueError(
            "No IMU data found in the recording. Cannot export motion data."
        ) from e

    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        motion_dir.mkdir(parents=True)
    if motion_dir.name != "motion":
        raise RuntimeWarning(
            f"Directory name {motion_dir.name} is not 'motion' as specified by Motion-BIDS"
        )
    if prefix is None:
        prefix = f"sub-{rec.info['wearer_name']}_task-TaskName_tracksys-NeonIMU"
    for field in ["sub-", "task-", "tracksys-"]:
        if field not in prefix:
            raise ValueError(f"Prefix must contain '{field}<label>' field.")

    # Define file paths
    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    channels_tsv_path = motion_dir / f"{prefix}_channels.tsv"
    channels_json_path = motion_dir / f"{prefix}_channels.json"

    # Export IMU data
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
    metadata = MOTION_META_DEFAULT.copy()
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


def export_eye_bids(
    rec: "Recording",
    output_dir: str | Path,
    prefix: Optional[str] = None,
    extra_metadata: dict = {},
):
    """ """
    try:
        gaze = rec.gaze
    except Exception as e:
        raise ValueError(
            "Gaze data cannot be loaded and thus cannot be exported to Eye-Tracking BIDS."
        ) from e
    try:
        eye_states = rec.eye_states
        eye_states = eye_states.interpolate(gaze.ts)
    except Exception:
        warn(
            "Eye states data cannot be loaded. Will export gaze data without pupil diameter information."
        )
        eye_states = None
    info = rec.info

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    if prefix is None:
        prefix = f"sub-{rec.info['wearer_name']}_task-TaskName"
    if "sub-" not in prefix:
        raise ValueError("Prefix must contain 'sub-<label>' field.")

    physio_tsv_path = output_dir / f"{prefix}_physio.tsv.gz"
    physio_json_path = output_dir / f"{prefix}_physio.json"
    physioevents_tsv_path = output_dir / f"{prefix}_physioevents.tsv.gz"
    physioevents_json_path = output_dir / f"{prefix}_physioevents.json"

    physio_col_names = EYE_META_DEFAULT["Columns"]
    if eye_states is None:
        physio_col_names = [
            col for col in physio_col_names if "pupil_diameter" not in col
        ]
    physio_data = pd.DataFrame(columns=physio_col_names)

    physio_data["timestamp"] = gaze.ts
    physio_data["x_coordinate"] = gaze["gaze x [px]"].values
    physio_data["y_coordinate"] = gaze["gaze y [px]"].values
    if eye_states is not None:
        physio_data["left_pupil_diameter"] = eye_states[
            "pupil diameter left [mm]"
        ].values
        physio_data["right_pupil_diameter"] = eye_states[
            "pupil diameter right [mm]"
        ].values
    physio_data.to_csv(
        physio_tsv_path,
        sep="\t",
        index=False,
        header=False,
        na_rep="n/a",
        compression="gzip",
    )

    physio_metadata = EYE_META_DEFAULT.copy()
    physio_metadata.update(
        {
            "SamplingFrequency": gaze.sampling_freq_effective,
            "DeviceSerialNumber": info["module_serial_number"],
            "SoftwareVersions": (
                f"App version: {info['app_version']}; "
                f"Pipeline version: {info['pipeline_version']}"
            ),
        }
    )
    physio_metadata.update(extra_metadata)
    with open(physio_json_path, "w") as f:
        json.dump(physio_metadata, f, indent=4)

    physioevents_col_names = EYE_EVENTS_META_DEFAULT["Columns"]
    physioevents_data = pd.DataFrame(columns=physioevents_col_names)
    for attr_name in ["blinks", "saccades", "fixations"]:
        try:
            events_data = getattr(rec, attr_name).data
            # Rename "start timestamp [ns]" to "onset"
            events_data = events_data.rename(columns={"start timestamp [ns]": "onset"})
            # Re-compute duration in seconds
            events_data["duration"] = events_data["duration [ms]"] / 1000.0
            # Add trial_type column with value "blink", "saccade", or "fixation"
            events_data = events_data[["onset", "duration"]]
            events_data["trial_type"] = attr_name[:-1]  # remove "s" at the end
            physioevents_data = pd.concat(
                [physioevents_data, events_data], ignore_index=True
            )
        except Exception:
            warn(
                f"Could not process events for {attr_name}, skipping exporting events for this attribute."
            )

    # Try to process messages if available, and add them to the physioevents data
    try:
        events = rec.events.data
        events = events.rename(columns={"timestamp [ns]": "onset", "name": "message"})
        events = events[["onset", "message"]]
        physioevents_data = pd.concat([physioevents_data, events], ignore_index=True)
    except Exception:
        warn("Could not process messages, skipping exporting messages.")

    physioevents_data = physioevents_data.sort_values(by="onset")
    physioevents_data.to_csv(
        physioevents_tsv_path,
        sep="\t",
        index=False,
        header=True,
        na_rep="n/a",
        compression="gzip",
    )

    physioevents_metadata = EYE_EVENTS_META_DEFAULT.copy()
    with open(physioevents_json_path, "w") as f:
        json.dump(physioevents_metadata, f, indent=4)
