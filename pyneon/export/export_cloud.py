import json
from pathlib import Path
from shutil import copy
from typing import TYPE_CHECKING
from warnings import warn

import pandas as pd

if TYPE_CHECKING:
    from ..recording import Recording


def export_cloud_format(recording: "Recording", target_dir: str | Path):
    """Export the recording in a cloud-compatible format.

    Parameters
    ----------
    recording : Recording
        The recording to export. ``recording.format`` must be "native".
    target_dir : str | Path
        The target directory to save the exported files.
        Will be created if it does not exist.
    """
    if recording.format != "native":
        raise ValueError("Recording is already in Cloud format; no export needed.")

    target_dir = Path(target_dir)
    if target_dir == recording.recording_dir:
        raise ValueError(
            "Target directory must be different from the native recording directory."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    data_exports = [
        ("gaze", "gaze.csv"),
        ("imu", "imu.csv"),
        ("eye_states", "3d_eye_states.csv"),
        ("fixations", "fixations.csv"),
        ("saccades", "saccades.csv"),
        ("blinks", "blinks.csv"),
        ("events", "events.csv"),
    ]
    for attr_name, filename in data_exports:
        _export_data(recording, attr_name, filename, target_dir)

    _export_scene_video(recording, target_dir)
    _export_template(recording, target_dir)
    _export_info(recording, target_dir)


def _export_data(recording, attr_name, filename, target_dir):
    try:
        attr = getattr(recording, attr_name)
    except FileNotFoundError:
        warn(f"Warning: '{attr_name}' data file not found in recording.")
        return
    data = attr.data.copy()
    # Make timestamp index a column again
    if data.index.name == "timestamp [ns]":
        data.reset_index(inplace=True, drop=False)
    else:
        data.reset_index(inplace=True, drop=True)
    # Append recording ID and section ID columns
    data["section id"] = recording.info.get("section_id", pd.NA)
    data["recording id"] = recording.recording_id
    cols = data.columns.tolist()
    data = data[cols[-2:] + cols[:-2]]  # Move new columns to front
    # Export to CSV
    data.to_csv(target_dir / filename, index=False)


def _export_scene_video(recording: "Recording", target_dir: Path):
    if not hasattr(recording, "scene_video"):
        warn("Warning: 'scene_video' not found in recording.")
        return
    scene_video = recording.scene_video
    target_video_path = target_dir / scene_video.video_file.name
    try:
        copy(scene_video.video_file, target_video_path)
    except Exception as e:
        warn(f"Warning: Failed to copy video file: {e}")
    # Export timestamps
    world_ts_df = pd.DataFrame(
        {
            "section_id": recording.info.get("section_id", pd.NA),
            "recording_id": recording.recording_id,
            "timestamp [ns]": scene_video.ts,
        }
    )
    world_ts_df.to_csv(target_dir / "world_timestamps.csv", index=False)
    with open(target_dir / "scene_camera.json", "w", encoding="utf-8") as f:
        json.dump(scene_video.info, f, indent=4)


def _export_template(recording, target_dir):
    if hasattr(recording, "recording_dir"):
        template_path = recording.recording_dir / "template.json"
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_json = json.load(f)
            if "items" in template_json and isinstance(template_json["items"], list):
                relevant_info_df = pd.DataFrame(template_json["items"])
                relevant_info_df.to_csv(target_dir / "template.csv", index=False)
            else:
                print("Warning: 'items' not found or not a list in template.json.")
        except Exception as e:
            print(f"Warning: Failed to read template.json: {e}")
    else:
        print("Warning: 'recording_dir' not found in recording.")


def _export_info(recording, target_dir):
    info = recording.info
    with open(target_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
