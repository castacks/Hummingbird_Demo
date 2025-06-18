import numpy as np
import cv2
from scipy.stats import circmean
from scipy.signal import find_peaks
import yaml

from . import wire_detection_utils as wdu

from . import ransac_bindings as rb

class WireDetector:
    def __init__(self, wire_detection_config_path, camera_intrinsics):

        with open(wire_detection_config_path, 'r') as file:
            wire_detection_config = yaml.safe_load(file)

        self.hough_vote_threshold = wire_detection_config['hough_vote_threshold']
        self.min_line_threshold = wire_detection_config['min_line_threshold']
        self.pixel_binning_size = wire_detection_config['pixel_binning_size']
        self.low_canny_threshold = wire_detection_config['low_canny_threshold']
        self.high_canny_threshold = wire_detection_config['high_canny_threshold']
        self.line_bin_avg_threshold_multiplier = wire_detection_config['line_bin_avg_threshold_multiplier']

        self.grad_bin_avg_threshold = wire_detection_config['grad_bin_avg_threshold']
        self.max_wire_per_roi = wire_detection_config['max_wire_per_roi']
        self.min_depth_clip = wire_detection_config['min_depth_clip_m']
        self.max_depth_clip = wire_detection_config['max_depth_clip_m']

        self.ransac_max_iters = wire_detection_config['ransac_max_iters']
        self.inlier_threshold_m = wire_detection_config['inlier_threshold_m']
        self.vert_angle_maximum_rad = wire_detection_config['vert_angle_maximum_rad']
        self.horz_angle_diff_maximum_rad = wire_detection_config['horz_angle_diff_maximum_rad']

        self.camera_intrinsics = camera_intrinsics
        self.inv_camera_intrinsics = np.linalg.inv(camera_intrinsics)

        self.img_height = None
        self.img_width = None
        self.img_shape = None
        self.cx = None
        self.cy = None
        self.line_length = None
        self.camera_rays = None
        
    # virtual method to be implemented by subclasses
    def get_hough_lines(self, rgb_image):
        pass

    def get_xy_depth_gradients(self, depth_image):
        pass

    # standard functions not depending on gpu or cpu
    def get_line_candidates(self, rgb_image):
        cartesian_lines = self.get_hough_lines(rgb_image)
        if cartesian_lines is None or len(cartesian_lines) == 0:
            return None, None, None

        line_angles = np.arctan2(
            cartesian_lines[:, 3] - cartesian_lines[:, 1],
            cartesian_lines[:, 2] - cartesian_lines[:, 0]
        )
        avg_angles = wdu.fold_angles_from_0_to_pi(line_angles)
        avg_angle = circmean(avg_angles, high=np.pi, low=-np.pi)
        if np.isnan(avg_angle) or np.isinf(avg_angle) or avg_angle == None:
            raise ValueError(f"Invalid average angle computed from line angles from {line_angles}.")

        if self.img_shape is None:
            self.img_shape = rgb_image.shape[:2]
            self.img_height, self.img_width = self.img_shape
            self.cx, self.cy = self.img_width // 2, self.img_height // 2
            self.line_length = max(self.img_width, self.img_height) * 2
            # Create meshgrid of pixel coordinates
            x_coords, y_coords = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))  # shape: (H, W)
            flatted_coord = np.column_stack((x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())))
            self.camera_rays = np.dot(self.inv_camera_intrinsics, flatted_coord.T).T        

        cos_avg, sin_avg = np.cos(avg_angle), np.sin(avg_angle)
        x0_avg = int(self.cx + self.line_length * cos_avg)
        y0_avg = int(self.cy + self.line_length * sin_avg)
        x1_avg = int(self.cx - self.line_length * cos_avg)
        y1_avg = int(self.cy - self.line_length * sin_avg)
        center_line = np.array([x0_avg, y0_avg, x1_avg, y1_avg])
        return cartesian_lines, center_line, avg_angle

    def get_line_instance_locations(self, cartesian_lines, center_line, avg_angle):
        perp_angle = wdu.perpendicular_angle_rad(avg_angle)
        image_perp_distance = wdu.get_length_of_center_line_across_image(self.img_height, self.img_width, perp_angle)
        bins = np.arange(- image_perp_distance // 2, image_perp_distance // 2 + 1, 1)

        pixel_dists_wrt_center = self.compute_perpendicular_distance(center_line, cartesian_lines)
        hist, bin_edges = np.histogram(pixel_dists_wrt_center, bins=bins)
        
        # find a threshold for where to count wire peaks based on count
        bin_threshold = self.line_bin_avg_threshold_multiplier * np.mean(hist[hist > 0])

        wire_distances_wrt_center = wdu.peak_hist_into_wires(hist, bin_edges, pixel_dists_wrt_center, bin_threshold) * -1 # flipping the sign to match the direction of the center line

        sin_offset, cos_offset = np.sin(avg_angle + np.pi / 2), np.cos(avg_angle + np.pi / 2)
        wire_midpoints = np.column_stack((
            self.cx - wire_distances_wrt_center * cos_offset,
            self.cy - wire_distances_wrt_center * sin_offset
        ))

        # Compute wire lines in a vectorized manner
        new_x0 = wire_midpoints[:, 0] + self.line_length * np.cos(avg_angle)
        new_y0 = wire_midpoints[:, 1] + self.line_length * np.sin(avg_angle)
        new_x1 = wire_midpoints[:, 0] - self.line_length * np.cos(avg_angle)
        new_y1 = wire_midpoints[:, 1] - self.line_length * np.sin(avg_angle)

        wire_lines = np.column_stack((new_x0, new_y0, new_x1, new_y1)).astype(int)

        return wire_lines, wire_midpoints, hist, bin_edges, bin_threshold, wire_distances_wrt_center
    
    def compute_perpendicular_distance(self, center_line, lines):
        """
        Computes the perpendicular distance from each line pixel to the center line.
        Parameters:
            center_line (np.ndarray): Coordinates of the center line in the format [x1, y1, x2, y2].
            lines (np.ndarray): Array of lines in the format [x1, y1, x2, y2].
        Returns:
            np.ndarray: Perpendicular distances from each line pixel to the center line.
        """
        x1, y1, x2, y2 = center_line
        
        # Compute coefficients A, B, C of the line equation Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # Extract pixel coordinates
        pixels = wdu.get_pixels_from_lines(lines, self.img_height, self.img_width)
        x0 = pixels[:, 0]
        y0 = pixels[:, 1]
        
        # Compute perpendicular distances
        distances = (A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        return distances
    
    def detect_wires_2d(self, rgb_image):
        cartesian_lines, center_line, avg_angle = self.get_line_candidates(rgb_image)
        if cartesian_lines is not None: 
            wire_lines, wire_midpoints, _ , _ , _ , midpoint_dists_wrt_center = self.get_line_instance_locations(cartesian_lines, center_line, avg_angle)
            wire_lines = np.array(wire_lines)
            wire_midpoints = np.array(wire_midpoints)
        else:
            wire_lines = np.array([])
            wire_midpoints = np.array([])
            midpoint_dists_wrt_center = np.array([])
            avg_angle = None
        return wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center
    
    def find_regions_of_interest(self, depth, avg_angle, midpoint_dists_wrt_center, viz_img=None):
        depth_gradient_x, depth_gradient_y = self.get_xy_depth_gradients(depth)
        perp_angle = wdu.perpendicular_angle_rad(avg_angle)
        depth_gradient = depth_gradient_x * np.cos(perp_angle) + depth_gradient_y * np.sin(perp_angle)

        distance, depth_gradient_1d = wdu.project_image_to_axis(depth_gradient, perp_angle)
        depth_gradient_1d = np.abs(depth_gradient_1d)
        depth_gradient_1d = depth_gradient_1d / np.max(depth_gradient_1d)

        dist_hist, bin_edges = np.histogram(distance, bins=np.arange(np.min(distance), np.max(distance), 1), weights=depth_gradient_1d)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        dist_hist = dist_hist / np.max(dist_hist)

        threshold = self.grad_bin_avg_threshold
        mask = dist_hist > threshold
        mask_diff = np.diff(mask.astype(int))
        mask_diff = np.concatenate(([0], mask_diff))

        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            return [], []

        if start_indices[0] > end_indices[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(mask) - 1)
        if len(end_indices) > len(start_indices):
            start_indices = np.insert(start_indices, 0, 0)

        assert len(start_indices) == len(end_indices), "Mismatch in start and end indices length"

        regions_of_interest = []
        roi_line_count = []
        for start, end in zip(start_indices, end_indices):
            if bin_centers[start] < bin_centers[end]:
                start = bin_centers[start]
                end = bin_centers[end]
                should_add_region = False
                line_count = 0
                for wire_dist in midpoint_dists_wrt_center: 
                    if start <= wire_dist <= end:
                        # Append the region to the list
                        line_count += 1
                        should_add_region = True
                        
                if should_add_region:
                    regions_of_interest.append((float(start), float(end)))
                    roi_line_count.append(int(line_count))

        # limit the count per region to the max number of wires per ROI
        roi_line_count = np.array(roi_line_count)
        roi_line_count = np.clip(roi_line_count, 0, self.max_wire_per_roi)
        return regions_of_interest, roi_line_count

    def roi_to_point_clouds(self, rois, avg_angle, depth_image, viz_img=None):
        """
        Convert a region of interest (ROI) to a binary mask.

        Parameters:
            roi (np.ndarray): Array of shape (N, 4) where each row is [x1, y1, x2, y2].
            img_shape (tuple): Shape of the image (height, width).

        Returns:
            np.ndarray: Binary mask of the same shape as the image.
        """

        img_shape = depth_image.shape[:2]  # (height, width)
        img_center = np.array([img_shape[1] // 2, img_shape[0] // 2])  # (x, y)
        if viz_img is not None:
            viz_mask = np.zeros(viz_img.shape[:2], dtype=np.uint8)

        perp_angle = wdu.perpendicular_angle_rad(avg_angle)  # Get the perpendicular angle in radians

        # Example angle in radians
        # Make sure avg_angle is in radians — if it's in degrees, convert with np.radians()
        roi_depths = []
        roi_rgb = []
        for start, end in rois:
            center_dist = 0.5 * (start + end)  # scalar, along direction of avg_angle
            length = abs(end - start)          # width of the ROI

            # Compute center offset in image coordinates
            dx = center_dist * np.cos(perp_angle)
            dy = center_dist * np.sin(perp_angle)
            center_coords = (img_center[0] + dx, img_center[1] + dy)

            # Define rectangle size: length along projected axis, large height perpendicular
            size = (length, img_shape[0] * 2)  # (width, height)

            # Create rotated rectangle
            rect = (center_coords, size, np.degrees(perp_angle))
            box = cv2.boxPoints(rect).astype(int)

            # Draw box on mask
            if viz_img is not None:
                cv2.fillConvexPoly(viz_mask, box, 255)

            single_roi_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.fillConvexPoly(single_roi_mask, box, 255)
            single_roi_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=single_roi_mask)
            if viz_img is not None:
                single_roi_rgb_image = cv2.bitwise_and(viz_img, viz_img, mask=single_roi_mask)
                roi_rgb.append(single_roi_rgb_image)
            roi_depths.append(single_roi_depth_image)

        if viz_img is not None:
            masked_viz_img = cv2.bitwise_and(viz_img, viz_img, mask=viz_mask)
            depth_img_masked = cv2.bitwise_and(depth_image, depth_image, mask=viz_mask)
            return roi_depths, depth_img_masked, roi_rgb, masked_viz_img
        else:
            return roi_depths, None, None, None
    
    def ransac_on_rois_cpp(self, rois, roi_line_counts, avg_angle, depth_image, viz_img=None):
        """
        Find wires in 3D from the regions of interest.

        Parameters:
            roi_depths (list): List of depth images for each ROI.
            roi_rgbs (list): List of RGB images for each ROI.
            avg_angle (float): Average angle of the wires in radians.
            roi_line_count (list): List of line counts for each ROI.

        Returns:
            fitted_lines (list): List of fitted lines in 3D.
        """
        depth_image = np.ascontiguousarray(depth_image, dtype=np.float32)
        assert depth_image.ndim == 2, "Depth image must be 2D"
        if viz_img is not None:
            masked_viz_img = np.ascontiguousarray(viz_img, dtype=np.uint8)
            assert masked_viz_img.ndim == 3 and masked_viz_img.shape[2] == 3, "Viz image must be 3D with 3 channels"
        else:
            masked_viz_img = None
        result = rb.ransac_on_rois(rois, roi_line_counts, 
                                        float(avg_angle), 
                                        self.camera_rays.astype(np.float32), 
                                        depth_image, 
                                        float(self.min_depth_clip),
                                        float(self.max_depth_clip), 
                                        int(self.ransac_max_iters), 
                                        float(self.inlier_threshold_m),
                                        float(self.vert_angle_maximum_rad),
                                        float(self.horz_angle_diff_maximum_rad), 
                                        viz_img=viz_img if viz_img is not None else None)

        pc_colors = result.point_colors
        if pc_colors is not None:
            for i in range(len(pc_colors)):
                if pc_colors[i] is not None:
                    pc_colors[i] = pc_colors[i].astype(np.uint8)
        return  result.fitted_lines, result.inlier_counts, result.roi_point_clouds, pc_colors if viz_img is not None else None, result.masked_viz_img if viz_img is not None else None

    def ransac_on_rois(self, rois, roi_line_counts, avg_angle, depth_image, viz_img=None):
        """
        Find wires in 3D from the regions of interest.

        Parameters:
            roi_depths (list): List of depth images for each ROI.
            roi_rgbs (list): List of RGB images for each ROI.
            avg_angle (float): Average angle of the wires in radians.
            roi_line_count (list): List of line counts for each ROI.

        Returns:
            fitted_lines (list): List of fitted lines in 3D.
        """
        fitted_lines = []
        line_inlier_counts = []
        roi_pcs = []
        roi_point_colors = []
        roi_depths, depth_img_masked, roi_rgbs, masked_viz_img = self.roi_to_point_clouds(rois, avg_angle, depth_image, viz_img=viz_img)
        if roi_rgbs is None:
            roi_rgbs = [None] * len(roi_depths)
        for roi_depth, roi_rgb, line_count in zip(roi_depths, roi_rgbs, roi_line_counts):
            # convert depth image to point cloud
            points, colors = self.depth_to_pointcloud(roi_depth, rgb=roi_rgb)
            roi_pcs.append(points)
            if colors is not None:
                colors = (np.array(colors) / 255.0)[:,::-1]
                roi_point_colors.append(colors)
            lines, line_inlier_count = rb.ransac_line_fitting(points.astype(np.float32), float(avg_angle), int(line_count), int(self.ransac_max_iters),
                                                float(self.inlier_threshold_m), float(self.vert_angle_maximum_rad), float(self.horz_angle_diff_maximum_rad))

            fitted_lines += lines
            line_inlier_counts += line_inlier_count

        return fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors if roi_point_colors else None, masked_viz_img if viz_img is not None else None

    def detect_3d_wires(self, rgb_image, depth_image, generate_viz = False):
        """
        Find wires in 3D from the RGB and depth images.
        """
        wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = self.detect_wires_2d(rgb_image)
        if len(wire_lines) == 0 or len(wire_midpoints) == 0 or avg_angle is None:
            return [], [], [], [], None

        regions_of_interest, roi_line_counts = self.find_regions_of_interest(depth_image, avg_angle, midpoint_dists_wrt_center)

        if generate_viz:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois_cpp(regions_of_interest, roi_line_counts, avg_angle, depth_image, viz_img=rgb_image)
        else:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois_cpp(regions_of_interest, roi_line_counts, avg_angle, depth_image, viz_img=None)

        return fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked
    
    def depth_to_pointcloud(self, depth_image, rgb=None):
        """
        Convert a depth image to a 3D point cloud.

        Parameters:
            depth_image (np.ndarray): 2D depth image.
            camera_intrinsics (np.ndarray): Camera intrinsic matrix.

        Returns:
            np.ndarray: 3D point cloud of shape (N, 3).
        """
        # Compute 3D points
        z_coords = depth_image.flatten()
        valid_mask = ~np.isnan(z_coords) & (z_coords > self.min_depth_clip) & (z_coords < self.max_depth_clip)
        points = self.camera_rays * z_coords.reshape(-1, 1)
        points = points[valid_mask]
        if rgb is not None:
            rgb = rgb.reshape(-1, 3)
            rgb = rgb[valid_mask]
        return points, rgb if rgb is not None else None