nominal_sampling_rates = {"gaze": 100, "imu": 110, "eye_states": 200}

expected_files_cloud = [
    "3d_eye_states.csv",
    "blinks.csv",
    "events.csv",
    "fixations.csv",
    "gaze.csv",
    "imu.csv",
    "info.json",
    "labels.csv",
    "saccades.csv",
    "scene_camera.json",
    "world_timestamps.csv",
]

data_types = {
    # Events
    "end timestamp [ns]": "int64",
    "duration [ms]": "Int64",
    # Gaze
    "point_x": float,  # native
    "point_y": float,  # native
    "gaze x [px]": float,
    "gaze y [px]": float,
    "worn": "Int32",
    "fixation id": "Int32",
    "blink id": "Int32",
    "azimuth [deg]": float,
    "elevation [deg]": float,
    # 3D eye states
    "pupil diameter left [mm]": float,
    "pupil diameter right [mm]": float,
    "eyeball center left x [mm]": float,
    "eyeball center left y [mm]": float,
    "eyeball center left z [mm]": float,
    "eyeball center right x [mm]": float,
    "eyeball center right y [mm]": float,
    "eyeball center right z [mm]": float,
    "optical axis left x": float,
    "optical axis left y": float,
    "optical axis left z": float,
    "optical axis right x": float,
    "optical axis right y": float,
    "optical axis right z": float,
    "eyelid angle top left [rad]": float,
    "eyelid angle bottom left [rad]": float,
    "eyelid angle top right [rad]": float,
    "eyelid angle bottom right [rad]": float,
    "eyelid aperture left [mm]": float,
    "eyelid aperture right [mm]": float,
    # IMU
    "angular_velocity_x": float,  # native
    "angular_velocity_y": float,  # native
    "angular_velocity_z": float,  # native
    "gyro x [deg/s]": float,
    "gyro y [deg/s]": float,
    "gyro z [deg/s]": float,
    "acceleration_x": float,  # native
    "acceleration_y": float,  # native
    "acceleration_z": float,  # native
    "acceleration x [g]": float,
    "acceleration y [g]": float,
    "acceleration z [g]": float,
    "roll [deg]": float,
    "pitch [deg]": float,
    "yaw [deg]": float,
    "quaternion_w": float,  # native
    "quaternion_x": float,  # native
    "quaternion_y": float,  # native
    "quaternion_z": float,  # native
    "quaternion w": float,
    "quaternion x": float,
    "quaternion y": float,
    "quaternion z": float,
    # Blinks
    # "blink id": "Int32",
    # Fixations
    "fixation x [px]": float,
    "fixation y [px]": float,
    # Saccades
    "saccade id": "Int32",
    "amplitude [px]": float,
    "amplitude [deg]": float,
    "mean velocity [px/s]": float,
    "peak velocity [px/s]": float,
    # Events
    "name": str,
    "type": str,
    # Detections
    "processed_frame_idx": "Int64",
    "frame_idx": "Int64",
    "tag_id": "Int32",
    "corners": object,
    "center": object,
    # Homographies
    "homography": object,
    # Gaze on surface
    "x_trans": float,
    "y_trans": float,
    # Fixations on surface
    "gaze x [surface coord]": float,
    "gaze y [surface coord]": float,
    # scanpath
    "fixations": object,
}
