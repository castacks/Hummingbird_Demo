import numpy as np
import cv2

import wire_detection.wire_detection_utils as wdu

class PositionKalmanFilter:
    def __init__(self, wire_tracking_config):
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

        self.kf_ids = np.array([]).reshape(0, 1)  # Store Kalman filter IDs for visualization
        self.kf_points = np.array([]).reshape(0, 2)  # Store distance perpendicular to wire angle, and height
        self.kf_covariances = np.array([]).reshape(0, 2, 2)
        self.kf_colors = np.array([]).reshape(0, 3)  # Store Kalman filter colors for visualization
        self.valid_counts = np.array([]).reshape(0, 1)  # Store valid counts for each Kalman filter point

        self.Q = np.zeros((2, 2)) # Process noise covariance
        self.R = np.zeros((2, 2)) # Measurement noise covariance
        self.Q[0, 0] = self.z_predict_cov ** 2  # Process noise for z
        self.Q[1, 1] = self.y_predict_cov ** 2  # Process
        self.R[0, 0] = self.z_measurement_cov ** 2  # Measurement noise for z
        self.R[1, 1] = self.y_measurement_cov ** 2  #

        self.P_init = self.R.copy() * self.inital_cov_multiplier # Initial estimate covariance to account for initial uncertainty

        self.max_kf_id = 0
        self.target_kf_id = None  # ID of the Kalman filter point to track

    def add_kf(self, y0, z0):
        """
        Add a new Kalman filter point for visualization.
        
        Parameters:
        - y0: Initial y-coordinate.
        - z0: Initial z-coordinate.
        """
        self.max_kf_id += 1
        self.kf_ids = np.vstack((self.kf_ids, np.array([[self.max_kf_id]])))
        self.kf_points = np.vstack((self.kf_points, np.array([y0, z0])))
        self.kf_covariances = np.vstack((self.kf_covariances, self.P_init.reshape(1, 2, 2)))
        self.kf_colors = np.vstack((self.kf_colors, self.generate_viz_color()))
        self.valid_counts = np.vstack((self.valid_counts, np.array([[self.valid_count]])))

    def remove_stale_kfs(self):
        """
        Remove a Kalman filter point by its ID.
        
        Parameters:
        - kf_id: ID of the Kalman filter point to remove.
        """
        idx = np.where(self.valid_counts < 0)
        if idx.size > 0:
            self.kf_ids = np.delete(self.kf_ids, idx, axis=0)
            self.kf_points = np.delete(self.kf_points, idx, axis=0)
            self.kf_covariances = np.delete(self.kf_covariances, idx, axis=0)
            self.kf_colors = np.delete(self.kf_colors, idx, axis=0)
            self.valid_counts = np.delete(self.valid_counts, idx, axis=0)
        return len(idx) # Return the number of removed Kalman filter points

    def initialize_kfs(self, camera_points, wire_yaw):
        assert camera_points.shape[1] == 3, f"camera_points shape: {camera_points.shape}"  # Ensure points are in (x, y, z) format, N x 3

        dists = camera_points[:, 0] * np.sin(wire_yaw) - camera_points[:, 1] * np.cos(wire_yaw)
        self.kf_points = np.hstack((dists.reshape(-1, 1), camera_points[:, 2].reshape(-1, 1)))
        self.kf_ids = np.arange(1, len(camera_points) + 1).reshape(-1, 1)
        self.kf_covariances = np.repeat(self.P_init.reshape(1, 2, 2), len(camera_points), axis=0)
        self.kf_colors = self.generate_viz_color(len(camera_points))
        self.valid_counts = np.ones((len(camera_points), 1), dtype=int)

    def predict(self, relative_H_transform, previous_wire_yaw, current_wire_yaw):
        """Predict the state and estimate covariance after a relative transformation."""
        assert relative_H_transform.shape == (4, 4), f"relative_H_transform shape: {relative_H_transform.shape}"  # Ensure transformation matrix is 4x4
        # update the kalman points
        xy_points = self.get_xys_from_dists(self.kf_points[:, 0], previous_wire_yaw)
        xyz_points = np.hstack((xy_points, self.kf_points[:, 1].reshape(-1, 1)))
        transformed_points = np.dot(relative_H_transform[:3, :3], xyz_points.T).T + relative_H_transform[:3, 3]
        assert transformed_points.shape == (self.kf_points.shape[0], 3), f"transformed_points shape: {transformed_points.shape}"  # Ensure transformed points are in (x, y, z) format
        new_kf_dists = self.get_dists_from_xys(transformed_points[:, :2], current_wire_yaw)
        self.kf_points[:, 0] = new_kf_dists
        self.kf_points[:, 1] = transformed_points[:, 2]

        # Update the covariance matrix
        

    def update(self, cam_points, wire_yaw):
        """Update the state estimate using a new measurement."""
        assert cam_points.shape[1] == 3, f"cam_points shape: {cam_points.shape}"  # Ensure points are in (x, y, z) format, N x 3
    
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        measured_dists = self.get_dists_from_xys(cam_points[:, :2], wire_yaw)
        dh_measured = np.hstack((measured_dists, cam_points[:, 2].reshape(-1, 1)))  # Combine distances and heights
        assert dh_measured.shape[1] == 2, f"dh_measured shape: {dh_measured.shape}"  # Ensure points are in (distance, height) format

        # Update the state estimate and covariance
        y = measured_pos - self.curr_pos
        assert y.shape == (2, 1), f"y shape: {y.shape}"
        self.curr_pos = self.curr_pos + K @ y
        assert self.curr_pos.shape == (2, 1), f"curr_pos shape: {self.curr_pos.shape}"
        self.P = (np.eye(2) - K) @ self.P
        self.valid_count += 1

    def get_dists_from_xys(xy_points, wire_yaw):
        dists = xy_points[:, 0] * np.sin(wire_yaw) - xy_points[:, 1] * np.cos(wire_yaw)
        return dists.reshape(-1, 1)
    
    def get_xys_from_dists(dists, wire_yaw):
        x = dists * -np.sin(wire_yaw)
        y = dists * np.cos(wire_yaw)
        return np.column_stack((x, y))
    
    def generate_viz_color(self, num_colors=1):
        """
        Generate a random color for visualization.
        Returns:
            A tuple representing the RGB color.
        """
        hue  = np.random.randint(0, 255, num_colors, dtype=np.uint8)
        saturation = np.random.randint(int(0.5 * 255), 255, num_colors, dtype=np.uint8)
        value = np.ones(num_colors, dtype=np.uint8) * 255  # Full brightness
        return cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)[0][0]
    
class DirectionKalmanFilter:
    def __init__(self, direction0, wire_tracking_config):
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
        assert isinstance(direction0, (list, np.ndarray)), "direction0 must be a list or numpy array"
        assert len(direction0) == 3, "direction0 must have three elements (vx, vy, vz)"
        self.Q_val = wire_tracking_config['yaw_predict_covariance'] ** 2  # Process noise covariance
        self.R_val = wire_tracking_config['yaw_measurement_covariance'] ** 2  # Measurement noise covariance
        self.max_yaw_covariance = wire_tracking_config['max_yaw_covariance'] ** 2  # Maximum yaw covariance
        self.inital_cov_multiplier = wire_tracking_config['initial_yaw_covariance_multiplier']

        self.R = np.eye(3) * self.R_val  # Measurement noise covariance matrixq
        self.Q = np.eye(3) * self.Q_val  # Process noise covariance matrix

        self.curr_direction = np.array(direction0).reshape(-1, 1)  # Current direction vector
        self.P = self.inital_cov_multiplier * self.R  # Initial estimate covariance

    def predict(self, relative_rotation):
        """
        Predict the state and estimate covariance.
        """
        assert relative_rotation.shape == (3, 3), f"relative_rotation shape: {relative_rotation.shape}"  # Ensure rotation matrix is 3x3
        # State prediction
        self.curr_direction = relative_rotation @ self.curr_direction
        self.curr_direction /= np.linalg.norm(self.curr_direction)  # Normalize the direction vector
        assert self.curr_direction.shape == (3, 1), f"curr_direction shape: {self.curr_direction.shape}"
        self.P = relative_rotation @ self.P @ relative_rotation.T + self.Q

    def update(self, measured_direction):
        """
        Update the state estimate using a new measurement.
        
        Parameters:
        - measured_yaw: float, yaw angle in radians
        """
        # Compute angle difference between measured and predicted yaw
        
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        measured_direction = np.array(measured_direction).reshape(-1, 1)
        assert measured_direction.shape == (3, 1), f"measured_direction shape: {measured_direction.shape}"
        y = measured_direction - self.curr_direction
        assert y.shape == (3, 1), f"y shape: {y.shape}"
        self.curr_direction = self.curr_direction + K @ y
        self.curr_direction /= np.linalg.norm(self.curr_direction)  # Normalize the direction vector
        assert self.curr_direction.shape == (3, 1), f"curr_direction shape: {self.curr_direction.shape}"

        self.P = (np.eye(3) - K) @ self.P
    
    def get_yaw(self):
        return np.arctan2(self.curr_direction[1, 0], self.curr_direction[0, 0])
