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
    "SamplingFrequency": "",
    "ACCELChannelCount": 3,
    "GYROChannelCount": 3,
    "MissingValues": "n/a",
    "MotionChannelCount": 13,
    "ORNTChannelCount": 7,
    "SubjectArtefactDescription": "",
    "TrackedPointsCount": 0,
    "TrackingSystemName": "IMU included in Neon",
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
    "pupil_size_left": {
        # Description adapted from https://docs.pupil-labs.com/neon/data-collection/data-format/#_3d-eye-states-csv
        "Description": "Physical diameter of the pupil of the left eye",
        "Units": "mm",
    },
    "pupil_size_right": {
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
