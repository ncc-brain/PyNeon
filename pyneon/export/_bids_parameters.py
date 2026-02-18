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
        "LongName": "Gaze position (x)",
        "Description": "Horizontal gaze position x-coordinate in the scene camera frame, measured from the top-left corner",
        "Units": "pixel",
    },
    "y_coordinate": {
        "LongName": "Gaze position (y)",
        "Description": "Vertical gaze position y-coordinate in the scene camera frame, measured from the top-left corner",
        "Units": "pixel",
    },
    "pupil_size_left": {
        "Description": "Physical diameter of the left eye pupil, measured in millimeters",
        "Units": "mm",
    },
    "pupil_size_right": {
        "Description": "Physical diameter of the right eye pupil, measured in millimeters",
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
        "Description": "Event duration in seconds",
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
