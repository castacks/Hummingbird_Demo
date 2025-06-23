import numpy as np
import cv2

class PositionKalmanFilter:
    def __init__(self, y0, z0, wire_tracking_config):
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
        self.z_predict_cov = wire_tracking_config['z_predict_covariance']
        self.z_measurement_cov = wire_tracking_config['z_measurement_covariance']
        self.z_max_covariance = wire_tracking_config['z_max_covariance']

        self.y_predict_cov = wire_tracking_config['y_predict_covariance']
        self.y_measurement_cov = wire_tracking_config['y_measurement_covariance']
        self.y_max_covariance = wire_tracking_config['y_max_covariance']

        self.inital_cov_multiplier = wire_tracking_config['initial_yaw_covariance_multiplier']


        self.viz_color = self.generate_viz_color()
        self.valid_count = 0
        self.Q = np.zeros((2, 2)) # Process noise covariance
        self.R = np.zeros((2, 2)) # Measurement noise covariance

        self.Q[0, 0] = self.z_predict_cov ** 2  # Process noise for z
        self.Q[1, 1] = self.y_predict_cov ** 2  # Process
        self.R[0, 0] = self.z_measurement_cov ** 2  # Measurement noise for z
        self.R[1, 1] = self.y_measurement_cov ** 2  #

        self.P = self.R.copy() * self.inital_cov_multiplier # Initial estimate covariance to account for initial uncertainty

        self.curr_pos = np.array([y0, z0]).reshape(-1, 1)
        self.valid_count = 1

    def predict(self):
        """Predict the state and estimate covariance."""
        # State prediction
        self.curr_pos = self.curr_pos
        # Covariance prediction
        self.P += self.Q

    def update(self, y, z):
        """Update the state estimate using a new measurement."""
        # Calculate Kalman Gain

        measured_pos = np.array([[y], [z]]).reshape(-1, 1)
        assert measured_pos.shape == (2, 1), f"measured_pos shape: {measured_pos.shape}"

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # Update the state estimate and covariance
        y = measured_pos - self.curr_pos
        assert y.shape == (2, 1), f"y shape: {y.shape}"
        self.curr_pos = self.curr_pos + K @ y
        assert self.curr_pos.shape == (2, 1), f"curr_pos shape: {self.curr_pos.shape}"
        self.P = (np.eye(2) - K) @ self.P
        self.valid_count += 1

    def generate_viz_color(self):
        """
        Generate a random color for visualization.
        Returns:
            A tuple representing the RGB color.
        """
        hue  = np.random.randint(0, 255)
        saturation = np.random.randint(int(0.5 * 255), 255)
        value = 255
        return cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)[0][0].tolist()

class YawKalmanFilter:
    def __init__(self, yaw0, wire_tracking_config):
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
        self.Q_val = wire_tracking_config['yaw_predict_covariance'] ** 2  # Process noise covariance
        self.R_val = wire_tracking_config['yaw_measurement_covariance'] ** 2  # Measurement noise covariance
        self.max_yaw_covariance = wire_tracking_config['max_yaw_covariance'] ** 2  # Maximum yaw covariance
        self.inital_cov_multiplier = wire_tracking_config['initial_yaw_covariance_multiplier']

        self.curr_state = np.array([np.cos(yaw0), np.sin(yaw0)]).reshape(-1, 1)
        self.valid_count = 1

        angle_jacobian = self.get_angle_jacobian_squared(yaw0)
        self.P = self.inital_cov_multiplier * self.R_val * angle_jacobian
        
    def get_angle_jacobian_squared(self, yaw): # effecitvely J @ J.T
        """
        Calculate the Jacobian of the angle.
        
        Returns:
        - angle_jacobian: Jacobian matrix of the angle.
        """
        return np.array([[np.sin(yaw) ** 2, -np.sin(yaw) * np.cos(yaw)],
                         [-np.sin(yaw) * np.cos(yaw), np.cos(yaw) ** 2]])

    def predict(self, relative_yaw_change):
        """
        Predict the state and estimate covariance.
        """
        # State prediction
        F = np.array([[np.cos(relative_yaw_change), -np.sin(relative_yaw_change)],
                      [np.sin(relative_yaw_change), np.cos(relative_yaw_change)]])
        self.curr_state = F @ self.curr_state
        self.P = F @ self.P @ F.T + self.Q_val * self.get_angle_jacobian_squared(self.get_yaw())

    def update(self, measured_yaw):
        """
        Update the state estimate using a new measurement.
        
        Parameters:
        - measured_yaw: float, yaw angle in radians
        """
        # Compute angle difference between measured and predicted yaw
        predicted_yaw = self.get_yaw()
        dyaw = (measured_yaw - predicted_yaw)

        # Convert dyaw back to vector form (unit vector)
        residual = np.array([np.cos(dyaw), np.sin(dyaw)])

        # Kalman gain
        R = self.get_angle_jacobian_squared(measured_yaw) * self.R_val
        S = self.P + R
        K = self.P @ np.linalg.inv(S)

        # Update state
        self.curr_state = self.curr_state + K @ residual

        # Update covariance
        self.P = (np.eye(2) - K) @ self.P

    def update(self, measured_yaw):
        """
        Update the state estimate using a new measurement.
        
        Parameters:
        - z: Measurement vector.
        """
        # Calculate Kalman Gain
        R = self.get_angle_jacobian_squared(measured_yaw) * self.R_val
        S = self.P + R
        K = self.P @ np.linalg.inv(S)

        # Update the state estimate and covariance
        measured_yaw = np.array([np.cos(measured_yaw), np.sin(measured_yaw)])
        y = measured_yaw - self.curr_state
        self.curr_state = self.curr_state + K @ y
        self.P = (np.eye(2) - K) @ self.P
    
    def get_yaw(self):
        return np.arctan2(self.curr_state[1], self.curr_state[0])
