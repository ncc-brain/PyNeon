import numpy as np
import pandas as pd


def smooth_camera_pose(
    camera_position_raw: pd.DataFrame,
    initial_state_noise: float = 0.1,
    process_noise: float = 0.1,
    measurement_noise: float = 0.01,
    gating_threshold: float = 2.0,
    bidirectional: bool = False,
) -> pd.DataFrame:
    """
    Smooth camera position estimates using a Kalman filter with optional
    forward-backward smoothing (RTS smoother).

    This function handles missing measurements and propagates predictions across frames.

    Parameters
    ----------
    camera_position_raw : pandas.DataFrame
        DataFrame containing ``frame index`` and ``camera_pos`` columns.
    initial_state_noise : float, optional
        Initial state covariance scaling factor. Defaults to 0.1.
    process_noise : float, optional
        Process noise covariance scaling factor. Defaults to 0.1.
    measurement_noise : float, optional
        Measurement noise covariance scaling factor. Defaults to 0.01.
    gating_threshold : float, optional
        Mahalanobis distance threshold for gating outliers. Defaults to 2.0.
    bidirectional : bool, optional
        If True, applies forward-backward RTS smoothing. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with ``frame index`` and ``smoothed_camera_pos`` columns,
        or ``camera_pos`` if bidirectional is False.
    """

    state_dim = 3
    meas_dim = 3

    # Ensure the DataFrame is sorted by frame index
    camera_position_raw = camera_position_raw.sort_values("frame index")

    # Extract all frame indices and create a complete range
    all_frames = np.arange(
        camera_position_raw["frame index"].min(),
        camera_position_raw["frame index"].max() + 1,
    )

    # Create a lookup for frame detections
    position_lookup = dict(
        zip(camera_position_raw["frame index"], camera_position_raw["camera_pos"])
    )

    # Kalman filter matrices
    F = np.eye(state_dim)  # State transition: Identity
    H = np.eye(meas_dim)  # Measurement matrix: Identity
    Q = process_noise * np.eye(state_dim)  # Process noise covariance
    R = measurement_noise * np.eye(meas_dim)  # Measurement noise covariance

    # Forward pass storage
    x_fwd = []  # Forward state estimates
    P_fwd = []  # Forward covariances

    # Initialize
    x = np.array(position_lookup[all_frames[0]]).reshape(-1, 1)
    P = initial_state_noise * np.eye(state_dim)

    for frame in all_frames:
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q

        # Measurement update
        if frame in position_lookup:
            z = np.array(position_lookup[frame]).reshape(-1, 1)
            y = z - H @ x
            S = H @ P @ H.T + R
            d = np.sqrt((y.T @ np.linalg.inv(S) @ y).item())

            if d < gating_threshold:
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(state_dim) - K @ H) @ P

        x_fwd.append(x.copy())
        P_fwd.append(P.copy())

    # If bidirectional smoothing is not needed, return forward results
    if not bidirectional:
        smoothed_positions = [x.flatten() for x in x_fwd]
        result_df = pd.DataFrame(
            {"frame index": all_frames, "camera_pos": smoothed_positions}
        )
        return result_df

    # Backward pass (RTS Smoother)
    x_smooth = x_fwd.copy()
    P_smooth = P_fwd.copy()

    for k in reversed(range(len(all_frames) - 1)):
        decay_factor = (
            1.0 - (k / len(all_frames))
        ) ** 2  # Reduce smoothing at boundaries
        G = decay_factor * (P_fwd[k] @ F.T @ np.linalg.inv(P_fwd[k + 1]))
        x_smooth[k] = x_fwd[k] + G @ (x_smooth[k + 1] - F @ x_fwd[k])
        P_smooth[k] = P_fwd[k] + G @ (P_smooth[k + 1] - P_fwd[k + 1]) @ G.T

    # Final smoothed positions
    smoothed_positions = [x.flatten() for x in x_smooth]

    # Return results
    result_df = pd.DataFrame(
        {"frame index": all_frames, "camera_pos": smoothed_positions}
    )
    return result_df
