import json
from pathlib import Path

import pandas as pd
import pytest


class TestBIDS:
    """Tests for Motion-BIDS export functionality."""

    @pytest.mark.parametrize(
        "dataset_fixture",
        ["simple_dataset_native", "simple_dataset_cloud"],
    )
    @pytest.mark.parametrize(
        "extra_metadata",
        [
            {},
            {
                "TaskName": "LabMuse",
                "TaskDescription": "Watching artworks on the screen",
                "InstitutionName": "Streeling University",
                "InstitutionAddress": "Trantor, Galactic Empire",
                "InstitutionalDepartmentName": "Department of Psychohistory",
            },
        ],
    )
    def test_export_motion_bids_basic(
        self, request, dataset_fixture, extra_metadata, tmp_path
    ):
        """Test basic Motion-BIDS export with default prefix."""
        dataset = request.getfixturevalue(dataset_fixture)
        motion_dir = tmp_path / "motion"  # Doesn't need to be created
        not_motion_dir = tmp_path / "not_motion"

        for recording in dataset.recordings:
            imu = recording.imu.interpolate(max_gap_ms=None)
            recording.export_motion_bids(motion_dir, extra_metadata=extra_metadata)
            default_prefix = (
                f"sub-{recording.info['wearer_name']}_task-TaskName_tracksys-NeonIMU"
            )

            # Verify motion TSV structure
            motion_tsv_path = motion_dir / f"{default_prefix}_motion.tsv"
            motion_df = pd.read_csv(motion_tsv_path, sep="\t", header=None)
            assert motion_df.shape == (
                len(imu.data),
                len(imu.columns),
            )

            # Verify motion JSON metadata
            motion_json_path = motion_dir / f"{default_prefix}_motion.json"
            with open(motion_json_path, "r") as f:
                metadata = json.load(f)
            assert metadata["Manufacturer"] == "TDK InvenSense & Pupil Labs"
            assert metadata["TrackingSystemName"] == "Neon IMU"
            assert (
                metadata["DeviceSerialNumber"] == recording.info["module_serial_number"]
            )
            assert metadata["SoftwareVersions"] == (
                f"App version: {recording.info['app_version']}; "
                f"Pipeline version: {recording.info['pipeline_version']}"
            )
            assert metadata["SamplingFrequency"] == imu.sampling_freq_nominal
            if extra_metadata:
                for key, value in extra_metadata.items():
                    assert metadata[key] == value

            # Verify channels TSV structure
            channels_tsv_path = motion_dir / f"{default_prefix}_channels.tsv"
            channels_df = pd.read_csv(channels_tsv_path, sep="\t")
            required_columns = {
                "name",
                "component",
                "type",
                "tracked_point",
                "units",
                "sampling_frequency",
            }
            assert required_columns.issubset(set(channels_df.columns))
            assert len(channels_df) == len(imu.columns)
            assert all(channels_df["tracked_point"] == "Head")

        # Test invalid directory name
        for recording in dataset.recordings:
            with pytest.raises(
                RuntimeWarning,
                match="Directory name not_motion is not 'motion' as specified by Motion-BIDS",
            ):
                recording.export_motion_bids(not_motion_dir)

    @pytest.mark.parametrize(
        "dataset_fixture",
        ["simple_dataset_native", "simple_dataset_cloud"],
    )
    def test_export_motion_bids_with_prefix(self, request, dataset_fixture, tmp_path):
        """Test Motion-BIDS export with custom prefix."""
        dataset = request.getfixturevalue(dataset_fixture)
        valid_prefix_1 = "sub-01_ses-1_task-LabMuse_tracksys-NeonIMU_run-1"
        valid_prefix_2 = "sub-01_task-LabMuse_tracksys-NeonIMU"

        invalid_prefix_1 = "sub-01_task-LabMuse"
        missing_1 = "tracksys-"

        invalid_prefix_2 = "task-LabMuse_tracksys-NeonIMU"
        missing_2 = "sub-"

        invalid_prefix_3 = "sub-01_tracksys-NeonIMU"
        missing_3 = "task-"

        motion_dir = tmp_path / "motion"

        for recording in dataset.recordings:
            for valid_prefix in [valid_prefix_1, valid_prefix_2]:
                recording.export_motion_bids(motion_dir, prefix=valid_prefix)
            for invalid_prefix, missing_field in zip(
                [invalid_prefix_1, invalid_prefix_2, invalid_prefix_3],
                [missing_1, missing_2, missing_3],
            ):
                with pytest.raises(
                    ValueError,
                    match=f"Prefix must contain '{missing_field}<label>' field",
                ):
                    recording.export_motion_bids(motion_dir, prefix=invalid_prefix)

    @pytest.mark.parametrize(
        "dataset_fixture",
        ["simple_dataset_native", "simple_dataset_cloud"],
    )
    @pytest.mark.parametrize(
        "extra_metadata",
        [
            {},
            {
                "StartTime": 42.4242424242,
            },
        ],
    )
    def test_export_eye_tracking_bids_basic(
        self, request, dataset_fixture, extra_metadata, tmp_path
    ):
        """Test basic Eye-Tracking-BIDS export with default prefix."""
        dataset = request.getfixturevalue(dataset_fixture)

        output_dir = tmp_path

        for recording in dataset.recordings:
            try:
                _ = recording.blinks
                recording.export_eye_tracking_bids(
                    output_dir, extra_metadata=extra_metadata
                )
            except ValueError:
                with pytest.warns(
                    UserWarning,
                    match="Could not read blinks data. These events will not be exported.",
                ):
                    recording.export_eye_tracking_bids(
                        output_dir, extra_metadata=extra_metadata
                    )

            # Verify files are created
            prefix = f"sub-{recording.info['wearer_name']}_task-TaskName"

            # Verify physio TSV structure
            physio_tsv_path = output_dir / f"{prefix}_physio.tsv.gz"
            physio_df = pd.read_csv(
                physio_tsv_path, sep="\t", compression="gzip", header=None
            )
            assert physio_df.shape[0] == len(recording.gaze.data)
            assert physio_df.shape[1] >= 3  # At minimum timestamp and x/y gaze

            # Verify physio JSON metadata
            physio_json_path = output_dir / f"{prefix}_physio.json"
            with open(physio_json_path, "r") as f:
                metadata = json.load(f)
            assert metadata["Manufacturer"] == "Pupil Labs"
            assert metadata["ManufacturersModelName"] == "Neon"
            assert metadata["PhysioType"] == "eyetrack"
            assert (
                metadata["DeviceSerialNumber"] == recording.info["module_serial_number"]
            )
            assert "SoftwareVersions" in metadata
            assert "Columns" in metadata
            if "StartTime" in extra_metadata:
                assert metadata["StartTime"] == extra_metadata["StartTime"]

            # Verify physioevents TSV structure
            physioevents_tsv_path = output_dir / f"{prefix}_physioevents.tsv.gz"
            if physioevents_tsv_path.exists():
                physioevents_df = pd.read_csv(
                    physioevents_tsv_path, sep="\t", header=None, compression="gzip"
                )
                assert physioevents_df.shape[1] >= 2

    @pytest.mark.parametrize(
        "dataset_fixture",
        ["simple_dataset_native", "simple_dataset_cloud"],
    )
    def test_export_eye_tracking_bids_with_prefix(
        self, request, dataset_fixture, tmp_path
    ):
        """Test Eye-Tracking-BIDS export with custom prefix."""
        dataset = request.getfixturevalue(dataset_fixture)
        valid_prefix = "sub-01_ses-1_task-LabMuse"
        invalid_prefix = "task-test"  # Missing sub-

        output_dir = tmp_path

        for recording in dataset.recordings:
            try:
                _ = recording.blinks
                recording.export_eye_tracking_bids(output_dir, prefix=valid_prefix)
            except ValueError:
                with pytest.warns(
                    UserWarning,
                    match="Could not read blinks data. These events will not be exported.",
                ):
                    recording.export_eye_tracking_bids(output_dir, prefix=valid_prefix)

            # Test invalid prefix
            with pytest.raises(ValueError, match="Prefix must contain 'sub-<label>'"):
                recording.export_eye_tracking_bids(output_dir, prefix=invalid_prefix)
