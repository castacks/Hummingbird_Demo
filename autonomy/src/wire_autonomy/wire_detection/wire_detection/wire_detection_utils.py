import numpy as np
import cv2
from scipy.signal import find_peaks

def get_pixels_from_lines(lines, img_height, img_width):
    img_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(img_mask, (x1, y1), (x2, y2), 255, 1)
    pixels = np.argwhere(img_mask == 255)
    pixels = pixels[:, [1, 0]]  # Convert to (x, y) format
    return pixels

def peak_hist_into_wires(hist, bin_edges, distances_wrt_center, bin_threshold):
    """
    Computes average distances within histogram peaks whose counts exceed a threshold.

    Parameters:
        hist (np.ndarray): Histogram counts.
        bin_edges (np.ndarray): Histogram bin edges (length should be len(hist)+1).
        distances_wrt_center (np.ndarray): Original distances.
        bin_threshold (float): Minimum bin count to consider.

    Returns:
        np.ndarray: Filtered average distances for valid peaks.
    """
    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, height=bin_threshold)

    # Find the corresponding bin center for each peak
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    distances = bin_centers[peaks]

    return np.array(distances)

def get_length_of_center_line_across_image(image_height, image_width, angle):
    assert isinstance(image_height, int) and isinstance(image_width, int), "Image dimensions must be integers"
    assert isinstance(angle, (int, float)), "Angle must be a scalar"

    # Calculate the length of the center line across the image
    angle = fold_angles_from_0_to_pi(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    length = np.sqrt((image_height * sin_angle) ** 2 + (image_width * cos_angle) ** 2)
    return length

def perpendicular_angle_rad(angle_rad):
    try:
        perp_angle = fold_angles_from_0_to_pi(angle_rad + np.pi / 2)
        if perp_angle > np.pi / 2:
            perp_angle -= np.pi
        return perp_angle
    except Exception as e:
        raise ValueError(f"Invalid angle: {angle_rad}. Error: {e}")
    
def fold_angles_from_0_to_pi(angles):
    '''
    Fold angles to the range [0, π].
    '''
    angles = np.asarray(angles)  # Ensure input is an array
    angles = angles % (2 * np.pi)  # Wrap into [0, 2π)

    # Fold anything > π into [0, π]
    folded = np.where(angles > np.pi, angles - np.pi, angles)

    return folded.item() if np.isscalar(angles) else folded

def project_image_to_axis(value_img, yaw_rad):
    """
    Projects pixels from a depth image onto an axis through the image center at a given yaw angle,
    and pairs each projection with its corresponding depth value.

    Parameters:
        depth_image (np.ndarray): 2D depth image.
        yaw_degrees (float): Yaw angle in degrees.

    Returns:
        np.ndarray: Array of shape (N, 2) where each row is (normalized_projection, depth_value)
    """
    H, W = value_img.shape
    center = np.array([W / 2.0, H / 2.0])

    # Convert yaw to radians
    axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])

    # Create meshgrid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)
    coords = np.stack([x_coords - center[0], y_coords - center[1]], axis=2)  # shape: (H, W, 2)

    # Compute projections
    projections = (coords @ axis).flatten()  # shape: (H, W)

    values = value_img.flatten()
    return projections, values
 