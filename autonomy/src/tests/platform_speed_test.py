import numpy as np
import cv2
import time
import yaml

from wire_detector_platforms import WireDetectorCPU, WireDetectorGPU

with open('wire_detect_config.yaml', 'r') as file:
    detection_config = yaml.safe_load(file)

input_image_size = [480, 270] 
img = cv2.imread("test_img.png")
depth = np.load("1746650644535430624.npy")

img = cv2.resize(img, (input_image_size[0], input_image_size[1]))
depth = cv2.resize(depth, (input_image_size[0], input_image_size[1]))
assert img is not None, "Image not found"
assert depth is not None, "Depth image not found"

original_image_size = img.shape[:2][::-1]  # (width, height)

camera_intrinsics = np.eye(3)

print(f"Starting CPU wire detector...")
wire_detector_cpu = WireDetectorCPU(detection_config, camera_intrinsics)
print(f"Starting  GPU wire detector...")
wire_detector_gpu = WireDetectorGPU(detection_config, camera_intrinsics)

# Create segmentation mask
start_time_cpu = time.perf_counter()
wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector_cpu.detect_wires_2d(img)
end_time_cpu = time.perf_counter()
print(f"Time taken for CPU detection: {end_time_cpu - start_time_cpu:.6f} seconds")

start_time_gpu = time.perf_counter()
wire_lines_gpu, wire_midpoints_gpu, avg_angle_gpu, midpoint_dists_wrt_center_gpu = wire_detector_gpu.detect_wires_2d(img)
end_time_gpu = time.perf_counter()
print(f"Time taken for GPU detection: {end_time_gpu - start_time_gpu:.6f} seconds")

start_time_cpu = time.perf_counter()
regions_of_interest, roi_line_counts = wire_detector_cpu.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
end_time_cpu = time.perf_counter()
print(f"Time taken for CPU regions of interest: {end_time_cpu - start_time_cpu:.2f} seconds, {1 / (end_time_cpu - start_time_cpu):.2f} Hz")

start_time_gpu = time.perf_counter()
regions_of_interest_gpu, roi_line_counts_gpu = wire_detector_gpu.find_regions_of_interest(depth, avg_angle_gpu, midpoint_dists_wrt_center_gpu)
end_time_gpu = time.perf_counter()
print(f"Time taken for GPU regions of interest: {end_time_gpu - start_time_gpu:.2f} seconds, {1 / (end_time_gpu - start_time_gpu):.2f} Hz")

start_time_cpu = time.perf_counter()
fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector_cpu.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=img)
end_time_cpu = time.perf_counter()
print(f"Time taken for CPU RANSAC on ROIs : {end_time_cpu - start_time_cpu:.2f} seconds, {1 / (end_time_cpu - start_time_cpu):.2f} Hz")

start_time_gpu = time.perf_counter()
fitted_lines_gpu, line_inlier_counts_gpu, roi_pcs_gpu, roi_point_colors_gpu, rgb_masked_gpu = wire_detector_gpu.ransac_on_rois(regions_of_interest_gpu, roi_line_counts_gpu, avg_angle_gpu, depth, viz_img=img)
end_time_gpu = time.perf_counter()
print(f"Time taken for GPU RANSAC on ROIs : {end_time_gpu - start_time_gpu:.2f} seconds, {1 / (end_time_gpu - start_time_gpu):.2f} Hz")



