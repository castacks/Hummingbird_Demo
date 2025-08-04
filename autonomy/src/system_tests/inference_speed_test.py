import numpy as np
import cv2
import time
import yaml

from wire_detection.wire_detector_platforms import WireDetectorCPU, WireDetectorGPU
from ament_index_python.packages import get_package_share_directory

detection_config_path = get_package_share_directory('wire_detection') + '/config/wire_detection_config.yaml'

input_image_size = [480, 270]  # (width, height)
img = cv2.imread("test_img.png")
depth = np.load("1746650644535430624.npy")

img = cv2.resize(img, (input_image_size[0], input_image_size[1]))
depth = cv2.resize(depth, (input_image_size[0], input_image_size[1]))
assert img is not None, "Image not found"
assert depth is not None, "Depth image not found"

original_image_size = img.shape[:2][::-1]  # (width, height)

camera_intrinsics = np.array([[314.81170654296875, 0.0, 240.47877502441406],
                            [0.0, 314.81170654296875, 136.25741577148438],
                            [0.0, 0.0, 1.0]], dtype=np.float32)
iterations = 50
print(f"Inferencing {iterations} iterations on CPU for image of size {input_image_size[0]}x{input_image_size[1]}...")

print(f"Starting CPU wire detector...")
lines_detected = 0
detect_2d_time = 0
roi_time = 0
ransac_time = 0
total_time = 0
wire_detector_cpu = WireDetectorCPU(detection_config_path, camera_intrinsics)

for i in range(iterations):
    total_start_time = time.perf_counter()
    start_time_cpu = time.perf_counter()
    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector_cpu.detect_wires_2d(img)
    end_time_cpu = time.perf_counter()
    detect_2d_time += end_time_cpu - start_time_cpu
    
    start_time_cpu = time.perf_counter()
    regions_of_interest, roi_line_counts = wire_detector_cpu.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
    end_time_cpu = time.perf_counter()
    roi_time += end_time_cpu - start_time_cpu

    start_time_cpu = time.perf_counter()
    fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector_cpu.ransac_on_rois_cpp(regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=img)
    end_time_cpu = time.perf_counter()
    ransac_time += end_time_cpu - start_time_cpu
    total_time += end_time_cpu - total_start_time
    lines_detected += len(fitted_lines)

print(f"Time taken for CPU Full Ransac: {total_time / iterations:.6f} seconds, {1 / (total_time / iterations):.6f} Hz")
print(f"Time taken for CPU 2d wire detections: {detect_2d_time / iterations:.6f} seconds, {1 / (detect_2d_time / iterations):.6f} Hz")
print(f"Time taken for CPU regions of interest: {roi_time / iterations:.6f} seconds, {1 / (roi_time / iterations):.6f} Hz")
print(f"Time taken for CPU RANSAC on ROIs : {ransac_time / iterations:.6f} seconds, {1 / (ransac_time / iterations):.6f} Hz")
print(f"Avg Number of lines detected: {lines_detected / iterations:.2f}")

print(f"Starting  GPU wire detector...")
lines_detected = 0
detect_2d_time = 0
roi_time = 0
ransac_time = 0
total_time = 0
wire_detector_gpu = WireDetectorGPU(detection_config_path, camera_intrinsics)
for i in range(iterations):
    total_start_time = time.perf_counter()
    start_time_gpu = time.perf_counter()
    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector_gpu.detect_wires_2d(img)
    end_time_gpu = time.perf_counter()
    detect_2d_time += end_time_gpu - start_time_gpu
    
    start_time_gpu = time.perf_counter()
    regions_of_interest, roi_line_counts = wire_detector_gpu.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
    end_time_gpu = time.perf_counter()
    roi_time += end_time_gpu - start_time_gpu
    
    start_time_gpu = time.perf_counter()
    fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector_gpu.ransac_on_rois_cpp(regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=img)
    end_time_gpu = time.perf_counter()
    ransac_time += end_time_gpu - start_time_gpu
    total_time += end_time_gpu - total_start_time
    lines_detected += len(fitted_lines)

print(f"Time taken for GPU Full Ransac: {total_time / iterations:.6f} seconds, {1 / (total_time / iterations):.6f} Hz")
print(f"Time taken for GPU 2d wire detections: {detect_2d_time / iterations:.6f} seconds, {1 / (detect_2d_time / iterations):.6f} Hz")
print(f"Time taken for GPU regions of interest: {roi_time / iterations:.6f} seconds, {1 / (roi_time / iterations):.6f} Hz")
print(f"Time taken for GPU RANSAC on ROIs : {ransac_time / iterations:.6f} seconds, {1 / (ransac_time / iterations):.6f} Hz")
print(f"Avg Number of lines detected: {lines_detected / iterations:.2f}")




