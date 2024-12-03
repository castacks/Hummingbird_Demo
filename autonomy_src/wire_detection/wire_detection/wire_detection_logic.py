import numpy as np
import cv2

class WireDetector:
    def __init__(self, threshold=500):
        self.threshold = threshold
        self.img_height = None
        self.img_width = None
        self.img_shape = None
        self.cx = None
        self.cy = None
        self.line_length = None

    def get_hough_lines(self, seg_mask, threshold):
        seg_coords = np.argwhere(seg_mask==255)
        seg_coords = seg_coords[:, [1, 0]]

        cartesian_lines = cv2.HoughLinesP(seg_mask, 1, np.pi/180, threshold)
        if cartesian_lines is None:
            return None, None, None, None, None
        cartesian_lines = np.squeeze(cartesian_lines,axis=1)

        polar_lines = cv2.HoughLines(seg_mask, 1, np.pi/180, threshold)
        polar_lines = np.squeeze(polar_lines, axis=1)
        if polar_lines is None:
            return None, None, None, None, None

        avg_angle = np.mean(polar_lines[:,1]) + np.pi / 2
        if self.img_shape == None:
            self.img_shape = seg_mask.shape
            self.img_height, self.img_width = self.img_shape
            self.cx, self.cy = self.img_shape[1] // 2, self.img_shape[0] // 2
            self.line_length = max(self.img_shape[1], self.img_shape[0]) * 2

        x0_avg = int(self.cx + self.line_length * np.cos(avg_angle))
        y0_avg = int(self.cy + self.line_length * np.sin(avg_angle))
        x1_avg = int(self.cx - self.line_length * np.cos(avg_angle))
        y1_avg = int(self.cy - self.line_length * np.sin(avg_angle))
        center_line = np.array([x0_avg, y0_avg, x1_avg, y1_avg])
        center_line_midpoint = ((x0_avg + x1_avg) / 2, (y0_avg + y1_avg) / 2)
        return cartesian_lines, center_line, center_line_midpoint, avg_angle, seg_coords

    def get_line_instance_locations(self, cartesian_lines, center_line, center_line_midpoint, avg_angle, seg_coords):
        distances_wrt_center = compute_perpendicular_distance(center_line, cartesian_lines)
        line_distances = np.sqrt((cartesian_lines[:,2] - cartesian_lines[:,0])**2 + (cartesian_lines[:,3] - cartesian_lines[:,1])**2).astype(np.int32)
        max_dist = np.max(distances_wrt_center)
        min_dist = np.min(distances_wrt_center)
        wire_pixel_seperation = 10
        num_bins = int((max_dist - min_dist) // wire_pixel_seperation)
        if num_bins == 0:
            num_bins = 1
        hist, bin_edges = np.histogram(distances_wrt_center, bins=num_bins, weights=line_distances)
        wire_distances_wrt_center = []
        for i, counts in enumerate(hist):
            if counts > np.mean(hist):
                binned_wire_distances = distances_wrt_center[(distances_wrt_center >= bin_edges[i]) & (distances_wrt_center < bin_edges[i+1])]
                if len(binned_wire_distances) != 0:
                    avg_distance = np.mean(binned_wire_distances)
                    wire_distances_wrt_center.append(avg_distance)
        wire_distances_wrt_center = np.array(wire_distances_wrt_center)
        wire_distances_wrt_center = wire_distances_wrt_center[~np.isnan(wire_distances_wrt_center)]

        wire_lines = []
        wire_midpoints = []
        for i, distance in enumerate(wire_distances_wrt_center):
            new_midpoint = np.array([center_line_midpoint[0] + distance * np.cos(avg_angle + np.pi/2), center_line_midpoint[1] + distance * np.sin(avg_angle + np.pi/2)])
            closest_index = np.argmin(np.linalg.norm(seg_coords - new_midpoint, axis=1))
            closest_wire_pixel = seg_coords[closest_index,:]
            wire_midpoints.append(closest_wire_pixel)
            new_x0 = int(closest_wire_pixel[0] + self.line_length * np.cos(avg_angle))
            new_y0 = int(closest_wire_pixel[1] + self.line_length * np.sin(avg_angle))
            new_x1 = int(closest_wire_pixel[0] - self.line_length * np.cos(avg_angle))
            new_y1 = int(closest_wire_pixel[1] - self.line_length * np.sin(avg_angle))
            wire_lines.append([new_x0, new_y0, new_x1, new_y1])

        return wire_lines, wire_midpoints

    def detect_wires(self, seg_mask, threshold=500):
        cartesian_lines, center_line, center_line_midpoint, avg_angle, seg_coords = self.get_hough_lines(seg_mask, threshold)
        if cartesian_lines is not None: 
            wire_lines, wire_midpoints = self.get_line_instance_locations(cartesian_lines, center_line, center_line_midpoint, avg_angle, seg_coords)
            wire_lines = np.array(wire_lines)
            wire_midpoints = np.array(wire_midpoints)
        else:
            wire_lines = np.array([])
            wire_midpoints = np.array([])
        return wire_lines, wire_midpoints, avg_angle

def compute_perpendicular_distance(center_line, lines):
    x1, y1, x2, y2 = center_line
    
    # Coefficients of the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    distances = []
    
    for x3, y3, x4, y4 in lines:
        # Calculate the midpoint of the line segment for distance measurement
        midpoint = ((x3 + x4) / 2, (y3 + y4) / 2)
        
        # Compute the perpendicular distance from the midpoint to the center line
        x0, y0 = midpoint
        distance = (A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        distances.append(distance)
    return np.array(distances)

def find_closest_point_on_3d_line(line_midpoint, yaw, target_point):
    x0, y0, z0 = line_midpoint
    xt, yt, zt = target_point

    # Direction vector based on yaw
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    assert not np.dot(direction, direction) < 1e-6, "Direction vector is zero"
    t = np.dot(np.array([xt - x0, yt - y0, zt - z0]), direction) / np.dot(direction, direction)
    closest_point = np.array([x0, y0, z0]) + t * direction
    return closest_point


def clamp_angles_deg(angles):
    angles = np.array(angles)
    converted_angles = []
    for angle in angles:
        if angle < 0:
            # Adjust negative angles
            converted_angle = (angle + 180) % 360
        elif angle > 180:
            # Convert angles greater than 180
            converted_angle = angle - 180
        else:
            converted_angle = angle
        
        converted_angles.append(converted_angle)
    
    return converted_angles
    
def clamp_angles_pi(angles):
    is_scalar = np.isscalar(angles)
    if is_scalar:
        angles = np.array([angles])
    converted_angles = []
    for angle in angles:
        if angle < 0:
            # Adjust negative angles
            converted_angle = (angle + np.pi) % (2 * np.pi)
        elif angle > np.pi:
            # Convert angles greater than pi
            converted_angle = angle - np.pi
        else:
            converted_angle = angle
        
        converted_angles.append(converted_angle)
    if is_scalar:
        return converted_angles[0]
    else:
        return np.array(converted_angles)

def get_yaw_from_quaternion(x, y, z, w):
    """
    Calculate the yaw (rotation around the z-axis) from a quaternion.
    
    Parameters:
    x, y, z, w -- components of the quaternion
    
    Returns:
    yaw -- the yaw angle in radians
    """
    # Calculate the yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    # yaw = clamp_angles_rad(yaw)
    return yaw

def get_distance_between_3D_point(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
    point1, point2 -- the 3D points
    
    Returns:
    distance -- the Euclidean distance between the points
    """
    # Calculate the Euclidean distance
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return distance
