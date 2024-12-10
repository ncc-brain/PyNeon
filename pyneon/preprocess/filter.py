import numpy as np
import pandas as pd

from typing import Optional


def smooth_camera_positions(
    camera_position_raw: pd.DataFrame,
    state_dim: int = 3,
    meas_dim: int = 3,
    process_noise: float = 0.005,
    measurement_noise: float = 0.005,
    gating_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Apply a Kalman filter to smooth camera positions and gate outliers based on Mahalanobis distance.
    Expects a DataFrame containing 'frame_idx' and 'camera_pos' columns, where 'camera_pos' is a
    length-3 array-like object representing [x, y, z] coordinates.

    Parameters
    ----------
    camera_position_raw : pd.DataFrame
        DataFrame containing 'frame_idx' and 'camera_pos' columns.
    state_dim : int, optional
        Dimensionality of the state vector. Default is 3 (x, y, z).
    meas_dim : int, optional
        Dimensionality of the measurement vector. Default is 3 (x, y, z).
    process_noise : float, optional
        Process noise covariance scaling factor. Default is 0.005.
    measurement_noise : float, optional
        Measurement noise covariance scaling factor. Default is 0.005.
    gating_threshold : float, optional
        Mahalanobis distance threshold for gating outliers. Default is 3.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same 'frame_idx' as input and an additional column 'smoothed_camera_pos'
        containing the smoothed positions.
    """
    # Ensure the DataFrame is sorted by frame_idx
    camera_position_raw = camera_position_raw.sort_values('frame_idx')

    # Extract positions and frame indices
    positions = np.stack(camera_position_raw['camera_pos'].values)
    frame_indices = camera_position_raw['frame_idx'].values

    # Define Kalman filter matrices
    F = np.eye(state_dim)  # State transition: Identity
    H = np.eye(meas_dim)   # Measurement matrix: Identity

    Q = process_noise * np.eye(state_dim)  # Process noise covariance
    R = measurement_noise * np.eye(meas_dim)  # Measurement noise covariance

    # Initial state from first measurement
    x = positions[0].reshape(-1, 1)
    P = 0.1 * np.eye(state_dim)

    smoothed_positions = [x.flatten()]

    for i in range(1, len(positions)):
        # Prediction
        x = F @ x
        P = F @ P @ F.T + Q

        # Measurement
        z = positions[i].reshape(-1, 1)

        # Compute innovation (residual) and innovation covariance
        y = z - H @ x
        S = H @ P @ H.T + R

        # Mahalanobis distance for gating
        d = np.sqrt((y.T @ np.linalg.inv(S) @ y).item())

        if d < gating_threshold:
            # Update step
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(state_dim) - K @ H) @ P
        else:
            # Outlier detected, skip update
            # (You could log or count occurrences if needed)
            pass

        smoothed_positions.append(x.flatten())

    smoothed_positions = np.array(smoothed_positions)

    # Create a new DataFrame with smoothed results
    smoothed_df = pd.DataFrame({
        'frame_idx': frame_indices,
        'smoothed_camera_pos': list(smoothed_positions)
    })

    final_results = camera_position_raw.copy()
    final_results['smoothed_camera_pos'] = smoothed_df['smoothed_camera_pos'].values

    return final_results
