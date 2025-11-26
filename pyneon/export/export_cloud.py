import os
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..recording import Recording

def export_cloud_format(recording: "Recording",
                        target_dir: str | Path):
    """Export the recording in a cloud-compatible format."""
    os.makedirs(target_dir, exist_ok=True)

    data_exports = [
        ("gaze", "gaze.csv"),
        ("imu", "imu.csv"),
        ("eye_states", "3d_eye_states.csv"),
        ("fixations", "fixations.csv"),
        ("saccades", "saccades.csv"),
        ("blinks", "blinks.csv"),
        ("events", "events.csv"),
    ]
    for attr, filename in data_exports:
        _export_data(recording, attr, filename, target_dir)

    _export_scene_video(recording, target_dir)
    _export_template(recording, target_dir)
    _export_info(recording, target_dir)

def _export_data(recording, attr, filename, target_dir):
    obj = getattr(recording, attr, None)
    if obj and hasattr(obj, "data"):
        obj.data.to_csv(os.path.join(target_dir, filename), index=False)
    else:
        print(f"Warning: '{attr}.data' not found in recording.")

def _export_scene_video(recording, target_dir):
    scene_video = getattr(recording, "scene_video", None)
    if scene_video and hasattr(scene_video, "video_file") and scene_video.video_file:
        video_filename = os.path.basename(scene_video.video_file)
        target_video_path = os.path.join(target_dir, video_filename)
        if scene_video.video_file != target_video_path:
            try:
                import shutil
                shutil.copy(scene_video.video_file, target_video_path)
            except Exception as e:
                print(f"Warning: Failed to copy video file: {e}")
            if hasattr(scene_video, "ts"):
                world_ts_df = pd.DataFrame()
                world_ts_df["timestamp [ns]"] = scene_video.ts
                world_ts_df["section_id"] = recording.info.get("section_id", 0)
                world_ts_df["recording_id"] = recording.recording_id

                world_ts_df.to_csv(os.path.join(target_dir, "world_timestamps.csv"), index=False)
            else:
                print("Warning: 'scene_video.ts' not found in recording.")
            if hasattr(scene_video, "info"):
                import json
                with open(os.path.join(target_dir, "scene_camera.json"), "w", encoding="utf-8") as f:
                    json.dump(scene_video.info, f, indent=4)
        else:
            print("Warning: 'scene_camera.video_path' not found or is None.")

def _export_template(recording, target_dir):
    if hasattr(recording, "recording_dir"):
        template_path = recording.recording_dir / "template.json"
        try:
            import json
            with open(template_path, "r", encoding="utf-8") as f:
                template_json = json.load(f)
            if "items" in template_json and isinstance(template_json["items"], list):
                relevant_info_df = pd.DataFrame(template_json["items"])
                relevant_info_df.to_csv(os.path.join(target_dir, "template.csv"), index=False)
            else:
                print("Warning: 'items' not found or not a list in template.json.")
        except Exception as e:
            print(f"Warning: Failed to read template.json: {e}")
    else:
        print("Warning: 'recording_dir' not found in recording.")

def _export_info(recording, target_dir):
    info = getattr(recording, "info", None)
    if info:
        #drop manifest if present
        info.pop("manifest", None)

        #directly dump info dictionary to json
        import json
        with open(os.path.join(target_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
    else:
        print("Warning: 'info' not found in recording.")