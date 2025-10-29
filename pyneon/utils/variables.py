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
    "worn": "Int8",
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
    "gyro x [deg/s]": float,
    "gyro y [deg/s]": float,
    "gyro z [deg/s]": float,
    "acceleration x [g]": float,
    "acceleration y [g]": float,
    "acceleration z [g]": float,
    "roll [deg]": float,
    "pitch [deg]": float,
    "yaw [deg]": float,
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

native_to_cloud_column_map = {
    # Time
    "time": "timestamp [ns]",
    # Gaze
    "point_x": "gaze x [px]",
    "point_y": "gaze y [px]",
    # Eye states
    "pupil_diameter_left_mm": "pupil diameter left [mm]",
    "pupil_diameter_right_mm": "pupil diameter right [mm]",
    "eyeball_center_left_x": "eye ball center left x [mm]",
    "eyeball_center_left_y": "eye ball center left y [mm]",
    "eyeball_center_left_z": "eye ball center left z [mm]",
    "eyeball_center_right_x": "eyeball center right x [mm]",
    "eyeball_center_right_y": "eyeball center right y [mm]",
    "eyeball_center_right_z": "eyeball center right z [mm]",
    "optical_axis_left_x": "optical axis left x",
    "optical_axis_left_y": "optical axis left y",
    "optical_axis_left_z": "optical axis left z",
    "optical_axis_right_x": "optical axis right x",
    "optical_axis_right_y": "optical axis right y",
    "optical_axis_right_z": "optical axis right z",
    "eyelid_angle_top_left": "eyelid angle top left",
    "eyelid_angle_bottom_left": "eyelid angle bottom left",
    "eyelid_angle_top_right": "eyelid angle top right",
    "eyelid_angle_bottom_right": "eyelid angle bottom right",
    "eyelid_aperture_left_mm": "eyelid aperture left [mm]",
    "eyelid_aperture_right_mm": "eyelid aperture right [mm]",
    # IMU
    "angular_velocity_x": "gyro x [deg/s]",
    "angular_velocity_y": "gyro y [deg/s]",
    "angular_velocity_z": "gyro z [deg/s]",
    "acceleration_x": "acceleration x [g]",
    "acceleration_y": "acceleration y [g]",
    "acceleration_z": "acceleration z [g]",
    "quaternion_w": "quaternion w",
    "quaternion_x": "quaternion x",
    "quaternion_y": "quaternion y",
    "quaternion_z": "quaternion z",
}
