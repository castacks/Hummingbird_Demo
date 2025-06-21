#!/usr/bin/env python

import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from wire_interfaces.msg import WireDetections

import wire_tracking.coord_transforms as ct
from wire_tracking.kalman_filters import PositionKalmanFilter, YawKalmanFilter

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireTrackingNode(Node):
    def __init__(self):
        super().__init__('wire_tracking_node')
        self.set_params()

        self.position_kalman_filters = {}
        self.vis_colors = {}
        self.yaw_kalman_filter = None
        self.are_kfs_initialized = False
        self.max_kf_label = 0
        self.line_length = None
        self.tracked_wire_id = None
        self.total_iterations = 0

        # Transform from pose_cam to wire_cam
        # 180 rotation about y-axis, 0.216m translation in negative x-axis
        self.H_wire_cam_to_pose_cam = np.array([[np.cos(np.deg2rad(180)),  0.0, np.sin(np.deg2rad(180)), - 0.216],
                                              [         0.0,             1.0,           0.0,           0.0],
                                              [-np.sin(np.deg2rad(180)), 0.0, np.cos(np.deg2rad(180)), 0.0],
                                              [0.0, 0.0, 0.0, 1.0]])        

        # Subscribers
        self.initialized = False
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1)
        self.camera_pose_sub = self.create_subscription(Odometry, self.pose_sub_topic, self.camera_pose_callback, 1)

        self.wire_estimates_sub = Subscriber(self, WireDetections, '/wire_detections')
        self.wire_estimates_sub = Subscriber(self, WireDetections, '/wire_detections')

        # Publishers
        self.tracking_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 10)
        self.pose_viz_pub = self.create_publisher(PoseStamped, self.pose_viz_pub_topic, 10)

        self.get_logger().info("Wire Tracking Node initialized")

    def camera_pose_callback(self, data):
        if not self.initialized:
            return
        
        # Update camera pose if needed
        self.camera_pose = data.pose.pose
        self.get_logger().info(f"Camera pose updated: {self.camera_pose.position.x}, {self.camera_pose.position.y}, {self.camera_pose.position.z}")

    def camera_info_callback(self, data):
        if self.initialized:
            return
        
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.camera_vector = np.array([self.fx, self.fy, self.cx, self.cy])
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                    [0, self.fy, self.cy],
                                    [0, 0, 1]])
        self.initialized = True
        self.get_logger().info("Wire Tracking Node initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)

    def wire_estimates_callback(self, msg):
        if not self.initialized:
            return
        
        global_midpoints = ct.image_to_world_pose(wire_midpoints, corresponding_depths, pose, self.camera_vector)

        if self.are_kfs_initialized == False:
            self.initialize_kfs(global_midpoints, global_yaw)
        else:
            self.update_kfs(global_midpoints, global_yaw, pose)

        # self.debug_kfs(detected_midpoints=wire_midpoints, depth_midpoints=corresponding_depths)=
        if self.tracked_wire_id is None and self.total_iterations > self.target_start_threshold:
            min_distance = np.inf
            min_id = None
            for i, kf in self.position_kalman_filters.items():
                if kf.valid_count >= self.valid_threshold:
                    curr_pos = pose.position
                    distance = ct.get_distance_between_3D_points(kf.curr_pos, np.array([curr_pos.x, curr_pos.y, curr_pos.z]))
                    if distance < min_distance:
                        min_distance = distance
                        min_id = i
            self.tracked_wire_id = min_id

        if self.are_kfs_initialized:
            debug_image = self.draw_kfs_viz(pose, wire_lines)
        else:
            debug_image = None        
        self.total_iterations += 1
        return debug_image
    
    def transform_pose_cam_to_wire_cam(self, pose_cam_pose):
        # self.get_logger().info(f"Pose Cam Pose: {pose_cam_pose.position.x}, {pose_cam_pose.position.y}, {pose_cam_pose.position.z}")
        H_pose_cam_to_world = ct.pose_to_homogeneous(pose_cam_pose)
        H_wire_cam_to_world = self.H_wire_cam_to_pose_cam @ H_pose_cam_to_world
        # assert False, f"Pose to wire transform: {H_pose_cam_to_world}"
        # H_wire_cam_to_world = H_pose_cam_to_world @ self.H_wire_cam_to_pose_cam
        wire_cam_pose = ct.homogeneous_to_pose(H_wire_cam_to_world)
        return wire_cam_pose

    def initialize_kfs(self, global_midpoints, global_yaw):
        if not self.are_kfs_initialized:
            # on initialization it if there are are wires that are too close together, it will merge them
            consolidated_midpoints = []
            for i in range(global_midpoints.shape[0]):
                got_matched = False
                mid = global_midpoints[i,:]
                if len(consolidated_midpoints) == 0:
                    consolidated_midpoints.append(mid)
                    continue
                else:
                    for j, merged_mid in enumerate(consolidated_midpoints):
                        distance = np.linalg.norm(mid - merged_mid, ord=2)
                        if distance < self.distance_threshold:
                            consolidated_midpoints[j] = (mid + merged_mid) / 2
                            got_matched = True
                            break

                if not got_matched:
                    consolidated_midpoints.append(mid)

            for mid in consolidated_midpoints:
                self.add_kf(mid)

            self.yaw_kalman_filter = YawKalmanFilter(global_yaw, yaw_covariance=self.yaw_covariance)
            self.are_kfs_initialized = True
            self.get_logger().info("Kalman Filters Initialized")

    def add_kf(self, midpoint):
        self.max_kf_label += 1
        self.position_kalman_filters[self.max_kf_label] = PositionKalmanFilter(midpoint, pos_covariance=self.pos_covariance)

    def debug_kfs(self, detected_midpoints = None, depth_midpoints = None, yaw=None, pose=None):
        valid_counts = []
        for i, kf in self.position_kalman_filters.items():
            valid_counts.append(kf.valid_count)
        if detected_midpoints is not None:
            self.get_logger().info("Detected Midpoints: %s" % str(detected_midpoints))
        if depth_midpoints is not None:
            self.get_logger().info("Detected Depths: %s" % str(depth_midpoints))
        if yaw is not None:
            self.get_logger().info("Yaw: %s" % yaw)
        if pose is not None:
            self.get_logger().info("Pose: %s" % pose)
        self.get_logger().info("Num KFs: %s" % len(self.position_kalman_filters))
        self.get_logger().info("Valid Counts: %s" % str(valid_counts))

    def update_kfs(self, new_global_midpoints, global_yaw, pose):
        matched_kf_ids = []
        new_midpoints = []
        self.yaw_kalman_filter.update(global_yaw)
        for midpoint in new_global_midpoints:
            got_matched = False
            for kf_id, kf in self.position_kalman_filters.items():
                closest_point = wd.find_closest_point_on_3d_line(midpoint, self.yaw_kalman_filter.get_yaw(), kf.curr_pos.flatten())
                distance = np.linalg.norm(closest_point - kf.curr_pos.T)
                # if the closest point on the detected line is close enough to an existing Kalman Filter, update it
                if distance < self.distance_threshold:
                    kf.update(midpoint)
                    matched_kf_ids.append(kf_id)
                    got_matched = True
                    break

            if not got_matched:
                new_midpoints.append(midpoint)

        # Add new Kalman Filters for unmatched midpoints
        for midpoint in new_midpoints:
            self.add_kf(midpoint)

        # Remove Kalman Filters that have not been updated for a while
        kfs_to_remove = []
        for kf_id, kf in self.position_kalman_filters.items():
            if kf_id not in matched_kf_ids:
                kf_pos = (kf.curr_pos).T
                x_img, y_img = ct.world_to_image_pose(kf_pos, pose, self.camera_vector)
                if x_img > 0 and x_img < self.cx * 2 and y_img > 0 and y_img < self.cy * 2:
                    kf.valid_count -= 1
                if kf.valid_count < 0:
                    kfs_to_remove.append(kf_id)
        for kf_id in kfs_to_remove:
            if kf_id == self.tracked_wire_id:
                self.tracked_wire_id = None
                self.get_logger().info("Tracked wire was removed")
            self.position_kalman_filters.pop(kf_id)
            self.vis_colors.pop(kf_id)

    def draw_kfs_viz(self, img, pose_in_world, wire_lines=None):

        if wire_lines is not None:
            green = (0, 255, 0)
            for i in range(len(wire_lines)):
                x0, y0, x1, y1 = wire_lines[i]
                cv2.line(img, (x0, y0), (x1, y1), green, 1)

        # assumes a up, right, forward world frame
        kf_yaw = self.yaw_kalman_filter.get_yaw()
        for i, kf in self.position_kalman_filters.items():
            if kf.valid_count >= self.valid_threshold:
                kf_pos = kf.curr_pos.flatten()
                z0 = kf_pos[2] + 1.0 * np.cos(kf_yaw)
                y0 = kf_pos[1] + 1.0 * np.sin(kf_yaw)
                z1 = kf_pos[2] - 1.0 * np.cos(kf_yaw)
                y1 = kf_pos[1] - 1.0 * np.sin(kf_yaw)
                global_line_points = np.array([[kf_pos[0], y0, z0], [kf_pos[0], y1, z1], [kf_pos[0], kf_pos[1], kf_pos[2]]])
                assert global_line_points.shape == (3, 3), f"global line points should be (3, 3), got {global_line_points.shape}"
                line_xs, line_ys = ct.world_to_image_pose(global_line_points, pose_in_world, self.camera_vector)
                image_yaw = np.arctan2(line_ys[1] - line_ys[0], line_xs[1] - line_xs[0])
                x0 = int(line_xs[0] + self.line_length * np.cos(image_yaw))
                y0 = int(line_ys[0] + self.line_length * np.sin(image_yaw))
                x1 = int(line_xs[1] - self.line_length * np.cos(image_yaw))
                y1 = int(line_ys[1] - self.line_length * np.sin(image_yaw))
                x_mid = int(line_xs[2])
                y_mid = int(line_ys[2])
                # self.get_logger().info(f"global_line_points: {global_line_points[0,:]}, {global_line_points[1,:]}")
                cv2.line(img, (x0, y0), (x1, y1), self.vis_colors[i], 2)
                cv2.circle(img, (int(x_mid), int(y_mid)), 5, self.vis_colors[i], -1)

        if self.tracked_wire_id is not None:
            tracked_midpoint = self.position_kalman_filters[self.tracked_wire_id].curr_pos
            image_midpoint = ct.world_to_image_pose(tracked_midpoint.T, pose_in_world, self.camera_vector)
            cv2.ellipse(img, (int(image_midpoint[0]), int(image_midpoint[1])), (15, 15), 0, 0, 360, (0, 255, 0), 3)
    
        return img
        
    def set_params(self):
        try:
            # sub topics
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.declare_parameter('pose_sub_topic', rclpy.Parameter.Type.STRING)
            self.pose_sub_topic = self.get_parameter('pose_sub_topic').get_parameter_value().string_value

            self.declare_parameter('wire_2d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_2d_viz_pub_topic = self.get_parameter('wire_2d_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_pc_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_pc_pub_topic = self.get_parameter('depth_pc_pub_topic').get_parameter_value().string_value
            self.declare_parameter('wire_3d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_3d_viz_pub_topic = self.get_parameter('wire_3d_viz_pub_topic').get_parameter_value().string_value

            # KF parameters
            self.declare_parameter('max_distance_threshold', rclpy.Parameter.Type.DOUBLE)
            self.distance_threshold = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
            self.declare_parameter('min_valid_kf_count_threshold', rclpy.Parameter.Type.INTEGER)
            self.valid_threshold = self.get_parameter('min_valid_kf_count_threshold').get_parameter_value().integer_value
            self.declare_parameter('iteration_start_threshold', rclpy.Parameter.Type.INTEGER)
            self.target_start_threshold = self.get_parameter('iteration_start_threshold').get_parameter_value().integer_value
            self.declare_parameter('yaw_covariance', rclpy.Parameter.Type.DOUBLE)
            self.yaw_covariance = self.get_parameter('yaw_covariance').get_parameter_value().double_value
            self.declare_parameter('pos_covariance', rclpy.Parameter.Type.DOUBLE)
            self.pos_covariance = self.get_parameter('pos_covariance').get_parameter_value().double_value

        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    

def main():
    rclpy.init()
    node = WireTrackingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()