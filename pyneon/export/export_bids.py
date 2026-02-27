import datetime
import gzip
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from warnings import warn

import pandas as pd

from ._bids_parameters import (
    EYE_EVENTS_META_DEFAULT,
    EYE_META_DEFAULT,
    MOTION_CHANNEL_MAP,
    MOTION_META_DEFAULT,
)

if TYPE_CHECKING:
    from ..recording import Recording


def _infer_prefix_from_dir(rec, output_dir):
    # Infer sub and ses names from motion_dir
    sub_name = f"sub-{rec.info['wearer_name']}"
    ses_name = None
    parent_dir = output_dir.parent
    if parent_dir.name.startswith("sub-"):
        sub_name = parent_dir.name
        ses_name = None
    elif parent_dir.name.startswith("ses-"):
        ses_name = parent_dir.name
        if parent_dir.parent.name.startswith("sub-"):
            sub_name = parent_dir.parent.name
    return sub_name, ses_name


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
    imu = rec.imu.interpolate(max_gap_ms=None)

    motion_dir = Path(motion_dir)
    if not motion_dir.is_dir():
        motion_dir.mkdir(parents=True)
    if motion_dir.name != "motion":
        raise RuntimeWarning(
            f"Directory name {motion_dir.name} is not 'motion' as specified by Motion-BIDS"
        )

    # Infer sub and ses names from motion_dir
    sub_name, ses_name = _infer_prefix_from_dir(rec, motion_dir)
    sub_ses_name = f"{sub_name}_{ses_name}" if ses_name else sub_name

    # If prefix is not provided, construct it using the inferred sub and ses names
    if prefix is None:
        if ses_name is None:
            prefix = f"{sub_name}_task-TaskName_tracksys-NeonIMU"
        else:
            prefix = f"{sub_name}_{ses_name}_task-TaskName_tracksys-NeonIMU"

    # Check if required fields are in the prefix
    for field in ["sub-", "task-", "tracksys-"]:
        if field not in prefix:
            raise ValueError(f"Prefix must contain '{field}<label>' field.")
    task_name = re.search(r"task-([^_]+)", prefix).group(1)

    # Define file paths
    motion_tsv_path = motion_dir / f"{prefix}_motion.tsv"
    motion_json_path = motion_dir / f"{prefix}_motion.json"
    channels_tsv_path = motion_dir / f"{prefix}_channels.tsv"
    channels_json_path = motion_dir / f"{prefix}_channels.json"

    # Export IMU data
    imu.data.to_csv(motion_tsv_path, sep="\t", index=False, header=False, na_rep="n/a")

    ch_names = imu.columns
    ch_names = [re.sub(r"\s\[[^\]]*\]", "", ch) for ch in ch_names]  # Strip units

    def get_channel_metadata(ch_name):
        key = " ".join(ch_name.strip().lower().split())
        meta = MOTION_CHANNEL_MAP.get(key)
        if meta is None:
            warn(
                "Unrecognized IMU channel name for Motion-BIDS export: "
                f"'{ch_name}'. Using 'n/a' for component/type/units."
            )
            return "n/a", "n/a", "n/a"
        return meta["component"], meta["type"], meta["units"]

    # Channels REQUIRED data
    channels = pd.DataFrame({"name": ch_names})
    channels[["component", "type", "units"]] = channels["name"].apply(
        lambda x: pd.Series(get_channel_metadata(x))
    )
    channels["tracked_point"] = "Head"
    channels = channels[["name", "component", "type", "tracked_point", "units"]]

    # Channels RECOMMENDED and OPTIONAL data
    channels["placement"] = "Head-mounted Neon glasses"
    channels["sampling_frequency"] = float(imu.sampling_freq_effective)
    channels["status"] = "good"
    channels["status_description"] = None
    channels.to_csv(channels_tsv_path, sep="\t", index=False, na_rep="n/a")

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
    motion_metadata = MOTION_META_DEFAULT.copy()
    motion_metadata.update(
        {
            "TaskName": task_name,
            "ACCELChannelCount": sum(channels["type"] == "ACCEL"),
            "GYROChannelCount": sum(channels["type"] == "GYRO"),
            "ORNTChannelCount": sum(channels["type"] == "ORNT"),
            "SamplingFrequencyEffective": imu.sampling_freq_effective,
            "DeviceSerialNumber": info["module_serial_number"],
            "SoftwareVersions": (
                f"App version: {info['app_version']}; "
                f"Pipeline version: {info['pipeline_version']}"
            ),
            "SamplingFrequency": imu.sampling_freq_nominal,
        }
    )
    motion_metadata.update(extra_metadata)

    with open(motion_json_path, "w") as f:
        json.dump(motion_metadata, f, indent=4)

    scans_path = motion_dir.parent / f"{sub_ses_name}_scans.tsv"
    filename = [motion_dir.name + "/" + motion_tsv_path.name]
    motion_acq_time = datetime.datetime.fromtimestamp(imu.first_ts / 1e9).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    new_scan = pd.DataFrame.from_dict(
        {
            "filename": filename,
            "acq_time": [motion_acq_time],
        }
    )
    if scans_path.is_file():
        scans = pd.read_csv(scans_path, sep="\t")
        if scans.filename.isin(filename).any():
            return
        else:
            scans = pd.concat([scans, new_scan], ignore_index=True)
    else:
        scans = new_scan
    scans.to_csv(scans_path, sep="\t", index=False)


def export_eye_tracking_bids(
    rec: "Recording",
    output_dir: str | Path,
    prefix: Optional[str] = None,
    extra_metadata: dict = {},
):
    """
    Export eye-tracking data to Eye-Tracking-BIDS format. Gaze position, pupil data,
    and eye-tracking events are saved as physiology data with metadata.

    Parameters
    ----------
    rec : Recording
        Recording instance containing the eye-tracking data.
    output_dir : str or pathlib.Path
        Output directory to save the Eye-Tracking-BIDS formatted data.
    prefix : str, optional
        Prefix for the BIDS filenames (must include ``sub-<label>`` and ``task-<label>``).
        If not provided, the function will attempt to infer the prefix from existing
        files in the output directory. Defaults to "sub-``wearer_name``_task-TaskName"
        if no existing files are found.
    extra_metadata : dict, optional
        Additional metadata to include in the JSON files. Defaults to an empty dict.

    Notes
    -----
    Eye-Tracking-BIDS is an extension to the Brain Imaging Data Structure (BIDS) to
    standardize the organization of eye-tracking data for reproducible research.
    For more information, see
    https://bids-specification.readthedocs.io/en/stable/modality-specific-files/eyetracking.html.

    The function automatically detects and uses matching prefixes from existing files
    in the output directory, allowing seamless integration with other modalities (e.g., motion).
    """
    gaze = rec.gaze
    try:
        eye_states = rec.eye_states
        eye_states = eye_states.interpolate(gaze.ts)
    except Exception:
        warn("Could not read eye states data. Pupil diameter will not be exported.")
        eye_states = None
    info = rec.info

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Infer sub and ses names from output_dir
    sub_name, ses_name = _infer_prefix_from_dir(rec, output_dir)

    # If prefix is not provided, construct a default one
    if prefix is None:
        # Try using the inferred sub and ses names
        if ses_name is None:
            prefix = f"{sub_name}_task-TaskName"
        else:
            prefix = f"{sub_name}_{ses_name}_task-TaskName"
        # Or better, use prefix in existing files in the output_dir if available
        existing_file = list(output_dir.glob("sub-*"))
        if existing_file:
            prefix = "_".join(existing_file[0].stem.split("_")[:-1])

    if "sub-" not in prefix:
        raise ValueError("Prefix must contain 'sub-<label>' field.")
    task_name_match = re.search(r"task-([^_]+)", prefix)
    task_name = task_name_match.group(1) if task_name_match else ""

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
    if (
        eye_states is not None
        and "pupil diameter left [mm]" in eye_states.columns
        and "pupil diameter right [mm]" in eye_states.columns
    ):
        physio_data["left_pupil_diameter"] = eye_states[
            "pupil diameter left [mm]"
        ].values
        physio_data["right_pupil_diameter"] = eye_states[
            "pupil diameter right [mm]"
        ].values
    with gzip.GzipFile(
        fileobj=open(physio_tsv_path, "wb"), mode="wb", mtime=0, filename=""
    ) as f:
        physio_data.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
            na_rep="n/a",
        )

    physio_metadata = EYE_META_DEFAULT.copy()
    physio_metadata.update(
        {
            "TaskName": task_name,
            "SamplingFrequency": gaze.sampling_freq_effective,
            "DeviceSerialNumber": info["module_serial_number"],
            "SoftwareVersions": (
                f"App version: {info['app_version']}; "
                f"Pipeline version: {info['pipeline_version']}"
            ),
        }
    )
    if eye_states is None:
        physio_metadata.pop("left_pupil_diameter")
        physio_metadata.pop("right_pupil_diameter")
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
            if physioevents_data.empty:
                physioevents_data = events_data
            else:
                physioevents_data = pd.concat(
                    [physioevents_data, events_data], ignore_index=True
                )
        except Exception:
            warn(f"Could not read {attr_name} data. These events will not be exported.")

    # Try to process messages if available, and add them to the physioevents data
    try:
        events = rec.events.data
        events = events.rename(columns={"timestamp [ns]": "onset", "name": "message"})
        events = events[["onset", "message"]]
        if physioevents_data.empty:
            physioevents_data = events
        else:
            physioevents_data = pd.concat(
                [physioevents_data, events], ignore_index=True
            )
    except Exception:
        warn("Could not read messages data. Messages will not be exported.")

    physioevents_data = physioevents_data.sort_values(by="onset")
    with gzip.GzipFile(
        fileobj=open(physioevents_tsv_path, "wb"), mode="wb", mtime=0, filename=""
    ) as f:
        physioevents_data.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
            na_rep="n/a",
        )

    physioevents_metadata = EYE_EVENTS_META_DEFAULT.copy()
    physioevents_metadata.update(
        {
            "TaskName": task_name,
        }
    )
    with open(physioevents_json_path, "w") as f:
        json.dump(physioevents_metadata, f, indent=4)
