import numpy as np
import pandas as pd

from typing import Optional

def smooth_camera_pose(
    camera_position_raw: pd.DataFrame,
    state_dim: int = 3,
    meas_dim: int = 3,
    initial_state_noise: float = 0.1,
    process_noise: float = 0.1,
    measurement_noise: float = 0.01,
    gating_threshold: float = 2.0,
    bidirectional: bool = False,
) -> pd.DataFrame:
    """
    Apply a Kalman filter to smooth camera positions, with optional forward-backward smoothing (RTS smoother).
    Handles missing measurements and propagates predictions.

    Parameters
    ----------
    camera_position_raw : pd.DataFrame
        DataFrame containing 'frame_idx' and 'camera_pos' columns.
    state_dim : int, optional
        Dimensionality of the state vector. Default is 3 (x, y, z).
    meas_dim : int, optional
        Dimensionality of the measurement vector. Default is 3 (x, y, z).
    initial_state_noise : float, optional
        Initial state covariance scaling factor. Default is 0.1.
    process_noise : float, optional
        Process noise covariance scaling factor. Default is 0.005.
    measurement_noise : float, optional
        Measurement noise covariance scaling factor. Default is 0.005.
    gating_threshold : float, optional
        Mahalanobis distance threshold for gating outliers. Default is 3.0.
    bidirectional : bool, optional
        If True, applies forward-backward RTS smoothing. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'frame_idx' and 'smoothed_camera_pos'.
    """
    # Ensure the DataFrame is sorted by frame_idx
    camera_position_raw = camera_position_raw.sort_values("frame_idx")

    # Extract all frame indices and create a complete range
    all_frames = np.arange(
        camera_position_raw["frame_idx"].min(),
        camera_position_raw["frame_idx"].max() + 1,
    )

    # Create a lookup for frame detections
    position_lookup = dict(
        zip(camera_position_raw["frame_idx"], camera_position_raw["camera_pos"])
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
            {"frame_idx": all_frames, "smoothed_camera_pos": smoothed_positions}
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
        {"frame_idx": all_frames, "smoothed_camera_pos": smoothed_positions}
    )
    return result_df
