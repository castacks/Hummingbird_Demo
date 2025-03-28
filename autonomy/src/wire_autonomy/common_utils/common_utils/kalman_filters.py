import numpy as np

class PositionKalmanFilter:
    def __init__(self, pos_0, pos_covariance=0.10):
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
        Q_val = pos_covariance ** 2 # 10 cm
        R_val = pos_covariance ** 2 # 10 cm
        self.Q = np.eye(3) * Q_val # Process noise covariance
        self.R = np.eye(3) * R_val # Measurement noise covariance
        self.P = np.eye(3) * R_val  # Initial estimate covariance
        assert len(pos_0) == 3
        self.curr_pos = np.array(pos_0).reshape(-1, 1) # Initial state estimate
        self.valid_count = 0     # Number of valid measurements to be considered a line

    def predict(self):
        """Predict the state and estimate covariance."""
        # State prediction
        self.curr_pos = self.curr_pos
        # Covariance prediction
        self.P += self.Q

    def update(self, measured_pos):
        """Update the state estimate using a new measurement."""
        # Calculate Kalman Gain

        measured_pos = np.array(measured_pos).reshape(-1, 1)
        assert measured_pos.shape == (3, 1), f"measured_pos shape: {measured_pos.shape}"

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # Update the state estimate and covariance
        y = measured_pos - self.curr_pos
        assert y.shape == (3, 1), f"y shape: {y.shape}"
        self.curr_pos = self.curr_pos + K @ y
        assert self.curr_pos.shape == (3, 1), f"curr_pos shape: {self.curr_pos.shape}"
        self.P = (np.eye(3) - K) @ self.P
        self.valid_count += 1

class YawKalmanFilter:
    def __init__(self, initial_yaw, yaw_covariance=0.0349066):
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
        Q_val = yaw_covariance ** 2 # 2 degrees
        R_val = yaw_covariance ** 2 # 2 degrees
        self.Q = np.eye(2) * Q_val # Process noise covariance in radians
        self.R = np.eye(2) * R_val # Measurement noise covariance in radians
        self.P = np.eye(2) * R_val  # Initial estimate covariance

        self.curr_state = np.array([np.cos(initial_yaw), np.sin(initial_yaw)])# Initial state estimate

    def predict(self):
        """
        Predict the state and estimate covariance.
        """
        # State prediction
        self.curr_state = self.curr_state
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
        K = self.P @ np.linalg.inv(S)

        # Update the state estimate and covariance
        measured_yaw = np.array([np.cos(measured_yaw), np.sin(measured_yaw)])
        y = measured_yaw - self.curr_state
        self.curr_state = self.curr_state + K @ y
        self.P = (np.eye(2) - K) @ self.P
    
    def get_yaw(self):
        return np.arctan2(self.curr_state[1], self.curr_state[0])
