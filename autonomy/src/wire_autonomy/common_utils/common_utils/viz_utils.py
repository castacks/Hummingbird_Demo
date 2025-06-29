import numpy as np
import cv2

def create_depth_viz(depth, min_depth, max_depth):
    depth[np.isnan(depth)] = min_depth
    depth[np.isinf(depth)] = min_depth
    depth[np.isneginf(depth)] = min_depth
    depth = np.clip(depth, min_depth, max_depth)

    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth

def draw_lines_on_image(image, lines, camera_intrinsics, color=(0, 255, 0), thickness=2):
    if lines is None or len(lines) == 0:
        return image

    line_points = np.array([]).reshape(0,3) # Nx3 array to hold line points
    for line in lines:
        start = line[0]
        end = line[1]
        line_points = np.append(line_points, start.reshape(1, 3), axis=0)
        line_points = np.append(line_points, end.reshape(1, 3), axis=0)
    # Project 3D points to 2D using camera intrinsics
    start_points_2d = camera_intrinsics @ line_points[::2, :3].T
    end_points_2d = camera_intrinsics @ line_points[1::2, :3].T
    start_points_2d = (start_points_2d[:2, :] / start_points_2d[2, :]).T
    end_points_2d = (end_points_2d[:2, :] / end_points_2d[2, :]).T
    start_points_2d = start_points_2d.astype(int)
    end_points_2d = end_points_2d.astype(int)
    for start, end in zip(start_points_2d, end_points_2d):
        cv2.line(image, tuple(start), tuple(end), color, thickness)
        
    return image