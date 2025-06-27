import numpy as np
import cv2

class PositionKalmanFilters:
    def __init__(self, wire_tracking_config, camera_intrinsics, image_size):
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
        self.z_predict_cov = wire_tracking_config['z_predict_covariance'] ** 2  # Process noise covariance for z, multiplied by 2 for better tracking
        self.z_measurement_cov = wire_tracking_config['z_measurement_covariance'] ** 2  # Process noise covariance for z, multiplied by 2 for better tracking
        self.z_max_covariance = wire_tracking_config['z_max_covariance'] ** 2 # Maximum covariance for z, multiplied by 2 for better tracking

        self.y_predict_cov = wire_tracking_config['y_predict_covariance'] ** 2  # Process noise covariance for y, multiplied by 2 for better tracking
        self.y_measurement_cov = wire_tracking_config['y_measurement_covariance'] ** 2  # Measurement noise covariance for y, multiplied by 2 for better tracking
        self.y_max_covariance = wire_tracking_config['y_max_covariance'] ** 2  # Maximum covariance for y, multiplied by 2 for better tracking

        self.inital_cov_multiplier = wire_tracking_config['initial_yaw_covariance_multiplier'] 
        self.wire_matching_min_threshold_m = wire_tracking_config['wire_matching_min_threshold_m']
        self.valid_count_buffer = wire_tracking_config['valid_count_buffer']  # Buffer for valid counts
        self.min_valid_kf_count_threshold = wire_tracking_config['min_valid_kf_count_threshold']  # Minimum valid count threshold for Kalman filter points

        self.camera_intrinsics = camera_intrinsics  # Store camera intrinsics for future use
        self.image_size = image_size

        self.initialized = False  # Flag to check if the Kalman filter is initialized

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

    def add_kfs(self, dists0, zs0):
        """
        Add multiple new Kalman filter points.
        
        Parameters:
        - dists0: Initial distance coordinates
        - z0: Initial z-coordinate.
        """
        if np.isscalar(dists0):
            dists0 = np.array([dists0])
        if np.isscalar(zs0):
            z0 = np.array([zs0])
        self.max_kf_id += 1
        new_ids = np.arange(self.max_kf_id, self.max_kf_id + len(dists0)).reshape(-1, 1)
        self.kf_ids = np.vstack((self.kf_ids, new_ids))
        self.kf_points = np.vstack((self.kf_points, np.column_stack((dists0, zs0))))
        self.kf_covariances = np.vstack((self.kf_covariances, np.tile(self.P_init, (len(dists0), 1, 1))))
        self.kf_colors = np.vstack((self.kf_colors, self.generate_viz_color(len(dists0))))
        self.valid_counts = np.vstack((self.valid_counts, np.full((len(dists0), 1), self.valid_count_buffer)))

    def remove_stale_kfs(self):
        """
        Remove a Kalman filter point by its ID.
        
        Parameters:
        - kf_id: ID of the Kalman filter point to remove.
        """
        idx = np.where(self.valid_counts < 0)
        target_id_index = np.where(self.kf_ids == self.target_kf_id)[0]
        if target_id_index.size > 0:
            self.target_kf_id = None  # Reset target ID if it is stale
        if idx.size > 0:
            self.kf_ids = np.delete(self.kf_ids, idx, axis=0)
            self.kf_points = np.delete(self.kf_points, idx, axis=0)
            self.kf_covariances = np.delete(self.kf_covariances, idx, axis=0)
            self.kf_colors = np.delete(self.kf_colors, idx, axis=0)
            self.valid_counts = np.delete(self.valid_counts, idx, axis=0)
        return len(idx) # Return the number of removed Kalman filter points

    def initialize_kfs(self, camera_points, wire_yaw):
        assert camera_points.shape[1] == 3, f"camera_points shape: {camera_points.shape}"  # Ensure points are in (x, y, z) format, N x 3
        dists = self.get_dists_from_xys(camera_points[:, :2], wire_yaw)
        self.add_kfs(dists.flatten(), camera_points[:, 2])  # Add Kalman filter points with distances and heights
        self.initialized = True  # Set the initialized flag to True

    def predict(self, relative_H_transform, previous_wire_yaw, current_wire_yaw):
        """Predict the state and estimate covariance after a relative transformation."""
        assert relative_H_transform.shape == (4, 4), f"relative_H_transform shape: {relative_H_transform.shape}"  # Ensure transformation matrix is 4x4
        # update the kalman points
        xy_points = self.get_xys_from_dists(self.kf_points[:, 0], previous_wire_yaw)
        xyz_points = np.hstack((xy_points, self.kf_points[:, 1].reshape(-1, 1)))
        transformed_points = np.dot(relative_H_transform[:3, :3], xyz_points.T).T + relative_H_transform[:3, 3]
        assert transformed_points.shape == (self.kf_points.shape[0], 3), f"transformed_points shape: {transformed_points.shape}"  # Ensure transformed points are in (x, y, z) format
        new_kf_dists = self.get_dists_from_xys(transformed_points[:, :2], current_wire_yaw)
        self.kf_points[:, 0] = new_kf_dists.flatten()
        self.kf_points[:, 1] = transformed_points[:, 2]

        # Update the covariance matrix for every Kalman filter point
        J = np.zeros((2, 2))
        c1 = np.cos(previous_wire_yaw)
        s1 = np.sin(previous_wire_yaw)
        c2 = np.cos(current_wire_yaw)
        s2 = np.sin(current_wire_yaw)
        R = relative_H_transform[:3, :3]
        J[0, 0] = s2 * (R[0,0] * s1 - R[0,1] * c1) + c2 * ( -R[1,0] * s1 + R[1,1] * c1)
        J[0, 1] = c2 * R[1,2] - s2 * R[0,2]
        J[1, 0] = -s1 * R[2,0] + c1 * R[2,1]
        J[1, 1] = R[2,2]
        assert J.shape == (2, 2), f"J shape: {J.shape}"  # Ensure Jacobian matrix is 2x2
        self.kf_covariances = J @ self.kf_covariances @ J.T + self.Q  # Update the covariance matrix  
        assert self.kf_covariances.shape == (self.kf_points.shape[0], 2, 2), f"kf_covariances shape: {self.kf_covariances.shape}"  # Ensure covariance matrix is in (N, 2, 2) format
    

    def update(self, cam_points, wire_yaw):
        """Update the state estimate using a new measurement."""
        assert cam_points.shape[1] == 3, f"cam_points shape: {cam_points.shape}"  # Ensure points are in (x, y, z) format, N x 3
    
        S = self.kf_covariances + self.R
        K = self.kf_covariances @ np.linalg.inv(S)

        measured_dists = self.get_dists_from_xys(cam_points[:, :2], wire_yaw)
        dh_measured = np.hstack((measured_dists, cam_points[:, 2].reshape(-1, 1)))  # Combine distances and heights
        assert dh_measured.shape[1] == 2, f"dh_measured shape: {dh_measured.shape}"  # Ensure points are in (distance, height) format

        comp_diffs = dh_measured[:, None, :] - self.kf_points[None, :, :] # gives a distance matrix of shape (N, M, 2) where N is the number of measured points and M is the number of Kalman filter points
        assert comp_diffs.shape == (dh_measured.shape[0], self.kf_points.shape[0], 2), f"comp_diffs shape: {comp_diffs.shape}"  # Ensure differences are in (kf_points, cam_points, 2) format

        comp_dists = np.linalg.norm(comp_diffs, axis=2)  # Compute
        matched_points = comp_dists < self.wire_matching_min_threshold_m  # Check if the minimum distance is below the threshold

        # Find Kalman filter points that were not matched, and decrement their valid counts
        unmatched_kfs = np.where(np.all(~matched_points, axis=0))[0]
        if unmatched_kfs.size > 0:
            kfs_in_frame = self.check_kfs_inframe(self.kf_points[unmatched_kfs, :], wire_yaw)
            valid_unmatched_indices = unmatched_kfs[kfs_in_frame]  # Get indices of Kalman filter points that are not in the current frame
            self.valid_counts[valid_unmatched_indices] -= 1  # Decrement the valid count for these Kalman filter points

        for i in range(matched_points.shape[0]): # iterate over each measured point
            if np.any(matched_points[i, :]):
                matched_index = np.where(matched_points[i, :]) 
                if len(matched_index) > 1:
                    raise ValueError(f"Multiple Kalman filter points matched for measured point {i}. This should not happen. Please check the matching logic.")
                matched_index = matched_index[0][0]  # Get the first matched index
                assert np.isscalar(matched_index), f"matched_index should be a scalar, got {matched_index}"  # Ensure matched_index is a scalar
                # update the Kalman filter point with the measurement
                y = dh_measured[i, :] - self.kf_points[matched_index, :]
                self.kf_points[matched_index, :] += (K[matched_index, :] @ y.T).T
                # Update the covariance matrix
                self.kf_covariances[matched_index, :, :] = (np.eye(2) - K[matched_index, :]) @ self.kf_covariances[matched_index, :]
                self.valid_counts[matched_index] += 1  # Increment the valid count for the Kalman filter point
            else: # if no matches add a new Kalman filter point
                self.add_kfs(dh_measured[i, 0], dh_measured[i, 1])
        
        # Remove stale Kalman filter points
        num_removed = self.remove_stale_kfs()
        if num_removed > 0:
            print(f"Removed {num_removed} stale Kalman filter points.")
                                                
    def get_dists_from_xys(self, xy_points, wire_yaw):
        dists = xy_points[:, 0] * -np.sin(wire_yaw) + xy_points[:, 1] * np.cos(wire_yaw)
        return dists.reshape(-1, 1)
    
    def get_xys_from_dists(self, dists, wire_yaw):
        x = dists * -np.sin(wire_yaw)
        y = dists * np.cos(wire_yaw)
        return np.column_stack((x, y))
    
    def check_kfs_inframe(self, kf_points, camera_yaw):
        """
        Check if a Kalman filter point is in the current frame.
        
        Parameters:
        - kf_id: ID of the Kalman filter point to check.
        
        Returns:
        - bool: True if the Kalman filter point is in the current frame, False otherwise.
        """
        xy_points = self.get_xys_from_dists(kf_points[:, 0], camera_yaw)
        xyz_points = np.hstack((xy_points, kf_points[:, 1].reshape(-1, 1)))  # Combine distances and heights
        image_points = np.dot(self.camera_intrinsics, xyz_points.T).T
        image_points /= image_points[:, 2][:, np.newaxis]  # Normalize by the third coordinate
        assert image_points.shape[1] == 3, f"image_points shape: {image_points.shape}"  # Ensure points are in (x, y, z)
        image_points = image_points[:, :2]  # Keep only (x, y)
        in_frame = image_points[:, 0] >= 0 and image_points[:, 0] < self.image_size[0] and image_points[:, 1] >= 0 and image_points[:, 1] < self.image_size[1]
        return in_frame
    
    def get_target_id(self):
        """
        Get the ID of the closest Kalman filter point.
        
        Returns:
        - int: ID of the closest Kalman filter point.
        """
        if self.target_kf_id is not None and self.target_kf_id in self.kf_ids:
            return self.target_kf_id
        
        if self.kf_ids.size == 0:
            return None
        
        else:
            valid_mask = self.valid_counts > self.valid_count_buffer
            if not np.any(valid_mask):
                return None
            else:
                valid_indices = np.where(valid_mask)[0]
                valid_heights = self.kf_points[valid_indices, 1]
                min_height_idx_within_valid = np.argmin(valid_heights)
                closest_index = valid_indices[min_height_idx_within_valid]
                self.target_kf_id = self.kf_ids[closest_index, 0]
        return self.target_kf_id
    
    def get_kf_by_id(self, kf_id):
        """
        Get the Kalman filter point by its ID.
        
        Parameters:
        - kf_id: ID of the Kalman filter point to get.
        
        Returns:
        - np.array: Kalman filter point in (distance, height) format.
        """
        if kf_id in self.kf_ids:
            index = np.where(self.kf_ids == kf_id)[0][0]
            return self.kf_points[index, :]
        else:
            raise ValueError(f"Kalman filter point with ID {kf_id} not found.")

    def generate_viz_color(self, num_colors=1):
        """
        Generate random colors for visualization using HSV -> BGR.
        Returns:
            An (N, 3) array representing RGB colors in uint8 format.
        """
        hue = np.random.randint(0, 180, num_colors, dtype=np.uint8)  # OpenCV uses H ∈ [0, 179]
        saturation = np.random.randint(int(0.5 * 255), 255, num_colors, dtype=np.uint8)
        value = np.full(num_colors, 255, dtype=np.uint8)  # Full brightness

        hsv_colors = np.stack((hue, saturation, value), axis=1)  # Shape: (N, 3)
        hsv_colors = hsv_colors[:, np.newaxis, :]  # Shape: (N, 1, 3) for cv2

        bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)
        rgb_colors = bgr_colors[:, 0, ::-1]  # Convert BGR to RGB and remove dummy dim

        return rgb_colors
    
class DirectionKalmanFilter:
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
        self.Q_val = wire_tracking_config['yaw_predict_covariance'] ** 2  # Process noise covariance
        self.R_val = wire_tracking_config['yaw_measurement_covariance'] ** 2  # Measurement noise covariance
        self.max_yaw_covariance = wire_tracking_config['max_yaw_covariance'] ** 2  # Maximum yaw covariance
        self.inital_cov_multiplier = wire_tracking_config['initial_yaw_covariance_multiplier']

        self.R = np.eye(3) * self.R_val  # Measurement noise covariance matrix
        self.Q = np.eye(3) * self.Q_val  # Process noise covariance matrix

        self.initialized = False  # Flag to check if the Kalman filter is initialized

    def initialize(self, direction0):
        """
        Initialize the Kalman Filter with a new direction.
        
        Parameters:
        - direction0: Initial direction vector as a list or numpy array.
        """
        assert isinstance(direction0, (list, np.ndarray)), "direction0 must be a list or numpy array"
        assert len(direction0) == 3, "direction0 must have three elements (vx, vy, vz)"
        self.curr_direction = np.array(direction0).reshape(-1, 1)
        self.P = self.inital_cov_multiplier * self.R  # Reset the covariance matrix
        self.initialized = True  # Set the initialized flag to True

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
        return self.get_yaw()  # Return the predicted yaw angle

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
    
    def get_direction_from_line_end_points(self, start, end, reference_direction=None):
        """
        Get the direction vector from two line end points.
        
        Parameters:
        - start: np.array, start point of the line
        - end: np.array, end point of the line
        
        Returns:
        - direction: np.array, normalized direction vector
        """
        assert start.shape == (3,) and end.shape == (3,), "start and end must be 3D points"
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        reference_direction = self.curr_direction if self.curr_direction is not None else reference_direction
        if np.dot(direction, reference_direction) < 0:
            direction = -direction
        
        return direction
    
    def get_direction(self):
        """
        Get the current direction vector.
        
        Returns:
        - direction: np.array, normalized direction vector
        """
        if not self.initialized:
            raise ValueError("Kalman filter is not initialized. Call initialize() first.")
        return self.curr_direction.flatten()

if __name__ == "__main__":
    # Example usage
    wire_tracking_config = {
        'z_predict_covariance': 0.1,
        'z_measurement_covariance': 0.05,
        'z_max_covariance': 0.5,
        'y_predict_covariance': 0.1,
        'y_measurement_covariance': 0.05,
        'y_max_covariance': 0.5,
        'initial_yaw_covariance_multiplier': 2.0,
        'wire_matching_min_threshold_m': 0.2,
        'valid_count_buffer': 3,  # Buffer for valid counts
        'yaw_predict_covariance': 0.1,
        'yaw_measurement_covariance': 0.05,
        'max_yaw_covariance': 0.5
    }
    camera_intrinsics = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Example camera intrinsics
    image_size = (640, 480)  # Example image size
    
    kf = PositionKalmanFilters(wire_tracking_config, camera_intrinsics, image_size)
    print("Kalman Filter initialized.")
    initial_cam_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    wire_yaw = np.pi / 4  # Example yaw angle
    kf.initialize_kfs(initial_cam_points, wire_yaw)
    print("Kalman Filters initialized with camera points and wire yaw.")
    relative_H_transform = np.eye(4)  # Identity transformation for simplicity
    previous_wire_yaw = wire_yaw
    current_wire_yaw = wire_yaw # Example new
    kf.predict(relative_H_transform, previous_wire_yaw, current_wire_yaw)
    print("Kalman Filters predicted state after transformation.")
    measured_cam_points = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [11.1, 12.1, 13.1]])
    kf.update(measured_cam_points, current_wire_yaw)
    print("Kalman Filters updated state with new measurements.")
    print("Current Kalman Filter points:", kf.kf_points)