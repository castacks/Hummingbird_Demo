import numpy as np

class PositionKalmanFilter:
    def __init__(self, pos_0):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        - F: State transition matrix.
        - H: Observation matrix.
        - Q: Process noise covariance.
        - R: Measurement noise covariance.
        - x0: Initial state estimate.
        - P0: Initial estimate covariance.
        """
        Q_val = 0.10 ** 2 # 10 cm
        R_val = 0.10 ** 2 # 10 cm
        self.Q = np.eye(3) * Q_val # Process noise covariance
        self.R = np.eye(3) * R_val # Measurement noise covariance
        self.P = np.eye(3) * R_val  # Initial estimate covariance

        self.curr_pos = pos_0         # Initial state estimate [x, y, z]
        self.valid_count = 0     # Number of valid measurements to be considered a line

        self.prev_predicted = pos_0

    def predict(self):
        """Predict the state and estimate covariance."""
        # State prediction
        self.curr_pos = self.curr_pos
        # Covariance prediction
        self.P += self.Q

    def update(self, measured_pos):
        """Update the state estimate using a new measurement."""
        # Calculate Kalman Gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # debug 
        # K = np.eye(3)

        # Update the state estimate and covariance
        y = measured_pos - self.curr_pos
        self.curr_pos = self.curr_pos + K @ y
        self.P = (np.eye(3) - K) @ self.P
        self.valid_count += 1

class YawKalmanFilter:
    def __init__(self, initial_yaw):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        - F: State transition matrix.
        - H: Observation matrix.
        - Q: Process noise covariance.
        - R: Measurement noise covariance.
        - x0: Initial state estimate.
        - P0: Initial estimate covariance.
        """
        Q_val = 0.0349066 ** 2 # 2 degrees
        R_val = 0.0349066 ** 2 # 2 degrees
        self.Q = np.eye(1) * Q_val # Process noise covariance in radians
        self.R = np.eye(1) * R_val # Measurement noise covariance in radians
        self.P = np.eye(1) * R_val  # Initial estimate covariance

        self.curr_yaw = initial_yaw         # Initial state estimate

        self.prev_predicted = initial_yaw

    def predict(self):
        """
        Predict the state and estimate covariance.
        """
        # State prediction
        self.curr_yaw = self.curr_yaw
        # Covariance prediction
        self.P += self.Q

    def update(self, measured_yaw):
        """
        Update the state estimate using a new measurement.
        
        Parameters:
        - z: Measurement vector.
        """
        # Calculate Kalman Gain
        S = self.P + self.R
        K = self.P / S

        # debug
        # K = np.eye(1)

        # Update the state estimate and covariance
        y = measured_yaw - self.curr_yaw
        self.curr_yaw += np.squeeze(K * y)

        self.P = (1 - K) * self.P
