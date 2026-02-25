from ..utils.variables import nominal_sampling_rates

MOTION_META_DEFAULT = {
    "TaskName": "",
    "TaskDescription": "",
    "Instructions": "",
    "DeviceSerialNumber": "",
    "Manufacturer": "TDK InvenSense & Pupil Labs",
    "ManufacturersModelName": "ICM-20948",
    "SoftwareVersions": "",
    "InstitutionName": "",
    "InstitutionAddress": "",
    "InstitutionalDepartmentName": "",
    "SamplingFrequency": nominal_sampling_rates["imu"],
    "ACCELChannelCount": 0,
    "ANGACCELChannelCount": 0,
    "GYROChannelCount": 0,
    "JNTANGChannelCount": 0,
    "LATENCYChannelCount": 0,
    "MAGNChannelCount": 0,
    "MISCChannelCount": 0,
    "MissingValues": "n/a",
    "MotionChannelCount": 0,
    "ORNTChannelCount": 0,
    "POSChannelCount": 0,
    "SamplingFrequencyEffective": nominal_sampling_rates["imu"],
    "SubjectArtefactDescription": "",
    "TrackedPointsCount": 0,
    "TrackingSystemName": "Neon IMU",
    "VELChannelCount": 0,
}

MOTION_CHANNEL_MAP = {
    "gyro x": {"component": "x", "type": "GYRO", "units": "deg/s"},
    "gyro y": {"component": "y", "type": "GYRO", "units": "deg/s"},
    "gyro z": {"component": "z", "type": "GYRO", "units": "deg/s"},
    "acceleration x": {"component": "x", "type": "ACCEL", "units": "g"},
    "acceleration y": {"component": "y", "type": "ACCEL", "units": "g"},
    "acceleration z": {"component": "z", "type": "ACCEL", "units": "g"},
    "roll": {"component": "x", "type": "ORNT", "units": "deg"},
    "pitch": {"component": "y", "type": "ORNT", "units": "deg"},
    "yaw": {"component": "z", "type": "ORNT", "units": "deg"},
    "quaternion w": {"component": "quat_w", "type": "ORNT", "units": "arbitrary"},
    "quaternion x": {"component": "quat_x", "type": "ORNT", "units": "arbitrary"},
    "quaternion y": {"component": "quat_y", "type": "ORNT", "units": "arbitrary"},
    "quaternion z": {"component": "quat_z", "type": "ORNT", "units": "arbitrary"},
}

EYE_META_DEFAULT = {
    "SamplingFrequency": "",
    "StartTime": 0,
    "Columns": [
        "timestamp",
        "x_coordinate",
        "y_coordinate",
        "left_pupil_diameter",
        "right_pupil_diameter",
    ],
    "DeviceSerialNumber": "",
    "Manufacturer": "Pupil Labs",
    "ManufacturersModelName": "Neon",
    "SoftwareVersions": "",
    "PhysioType": "eyetrack",
    "EnvironmentCoorinates": "top-left",
    "RecordedEye": "cyclopean",
    "SampleCoordinateSystem": "gaze-in-world",
    "EyeTrackingMethod": "real-time neural network",
    "timestamp": {
        "Description": "UTC timestamp in nanoseconds of the sample",
        "Units": "ns",
    },
    "x_coordinate": {
        # Description adapted from https://docs.pupil-labs.com/neon/data-collection/data-format/#gaze-csv
        "Description": "X-coordinate of the mapped gaze point in world camera pixel coordinates.",
        "Units": "pixel",
    },
    "y_coordinate": {
        "Description": "Y-coordinate of the mapped gaze point in world camera pixel coordinates.",
        "Units": "pixel",
    },
    "left_pupil_diameter": {
        # Description adapted from https://docs.pupil-labs.com/neon/data-collection/data-format/#_3d-eye-states-csv
        "Description": "Physical diameter of the pupil of the left eye",
        "Units": "mm",
    },
    "right_pupil_diameter": {
        "Description": "Physical diameter of the pupil of the right eye",
        "Units": "mm",
    },
}

EYE_EVENTS_META_DEFAULT = {
    "Columns": [
        "onset",
        "duration",
        "trial_type",
        "message",
    ],
    "Description": "Eye events and messages logged by Neon",
    "OnsetSource": "timestamp",
    "onset": {
        "Description": "UTC timestamp in nanoseconds of the start of the event",
        "Units": "ns",
    },
    "duration": {
        "Description": "Event duration",
        "Units": "s",
    },
    "trial_type": {
        "Description": "Type of trial event",
        "Levels": {
            "fixation": {
                "Description": "Fixation event",
            },
            "saccade": {
                "Description": "Saccade event",
            },
            "blink": {
                "Description": "Blink event",
            },
        },
    },
}
