#!/usr/bin/env python

import numpy as np
import cv2
import yaml
import os
import bisect

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from ament_index_python.packages import get_package_share_directory

from wire_interfaces.msg import WireDetections, WireTarget

import wire_tracking.coord_transforms as ct
from wire_tracking.kalman_filters import PositionKalmanFilters, DirectionKalmanFilter

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ZED_BASELINE_M = .12 / 2

class WireTrackingNode(Node):
    def __init__(self):
        super().__init__('wire_tracking_node')
        self.position_kalman_filters = None
        self.direction_kalman_filter = None
        self.set_params()

        self.bridge = CvBridge()
        self.previous_pose = None

        self.max_kf_label = 0
        self.total_iterations = 0

        self.tracked_wire_id = None

        # Transform from pose_cam to wire_cam
        # 180 rotation about y-axis, 0.216m translation in negative x-axis
        self.Rz = np.array([[np.cos(np.deg2rad(-90)), -np.sin(np.deg2rad(-90)), 0.0],
                            [np.sin(np.deg2rad(-90)), np.cos(np.deg2rad(-90)), 0.0],
                            [0.0, 0.0, 1.0]])
        self.Ry = np.array([[np.cos(np.deg2rad(180)), 0.0, np.sin(np.deg2rad(180))],
                            [0.0, 1.0, 0.0],
                            [-np.sin(np.deg2rad(180)), 0.0, np.cos(np.deg2rad(180))]])
        self.H_pose_to_wire = np.eye(4)
        self.H_pose_to_wire[:3, :3] = self.Rz @ self.Ry        
        self.H_pose_to_wire[:3, 3] = np.array([ZED_BASELINE_M, 0.0, -0.216]).T
        self.H_wire_to_pose = np.linalg.inv(self.H_pose_to_wire)

        # Subscribers
        self.initialized = False
        camera_info_callback_group = MutuallyExclusiveCallbackGroup()
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1, callback_group=camera_info_callback_group)

        pose_callback_group = ReentrantCallbackGroup()
        self.pose_sub = self.create_subscription(Odometry, self.pose_sub_topic, self.pose_callback, rclpy.qos.qos_profile_sensor_data, callback_group=pose_callback_group)
        self.relative_transform_queue = []  # Queue to hold relative pose transforms
        self.wire_detection_sub = self.create_subscription(WireDetections, self.wire_detections_pub_topic, self.wire_detection_callback, 1, callback_group=pose_callback_group)

        if self.wire_viz_2d:
            self.rgb_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.rgb_callback, rclpy.qos.qos_profile_sensor_data)
            self.rgb_timestamps = []
            self.rgb_imgs = []

        # Wire Publishers
        self.wire_target_pub = self.create_publisher(WireTarget, self.wire_target_pub_topic, 10)
        self.wire_target_pub_timer = self.create_timer(1.0 / self.target_publish_frequency_hz, self.target_timer_callback)

        # Visualization Publishers
        self.tracking_2d_pub = self.create_publisher(Image, self.tracking_2d_pub_topic, 1)
        self.tracking_3d_pub = self.create_publisher(Marker, self.tracking_3d_pub_topic, 1) 

        self.get_logger().info("Wire Tracking Node initialized")

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
        self.image_shape = (data.height, data.width)
        self.position_kalman_filters = PositionKalmanFilters(self.wire_tracking_config, self.camera_matrix, self.image_shape)
        self.get_logger().info("Wire Tracking Node initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)
        self.initialized = True

    def pose_callback(self, pose_msg):
        """
        Callback for the pose message. This is used to get the camera pose in the world frame.
        """
        if not self.initialized:
            return
        if self.previous_downward_pose is None:
            self.previous_downward_pose = pose_msg.pose
            return
        
        relative_pose_transform = ct.get_relative_transform(self.previous_downward_pose, pose_msg.pose)
        relative_wire_transform = self.get_relative_transform_in_wire_cam(relative_pose_transform)
        stamp = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9

        # Add the relative pose transform to the queue
        bisect.insort(self.relative_transform_queue, (stamp, relative_wire_transform))
        self.previous_downward_pose = pose_msg.pose

    def predict_kfs_up_to_current_pose(self, input_stamp):
        """
        Predict the Kalman filters up to the current pose timestamp.
        """
        if not self.initialized or not self.position_kalman_filters.initialized:
            self.get_logger().info("Wire Tracking Node not initialized yet. Waiting for camera info.")
            return
        if len(self.relative_transform_queue) == 0:
            self.get_logger().info("No relative pose transforms available. Waiting for pose updates.")
            return
        idx = bisect.bisect_right(self.relative_transform_queue, (input_stamp, None))

        poses = self.relative_transform_queue[:idx]
        if len(poses) == 0:
            self.get_logger().info("No relative pose transforms available for prediction.")
            return
        
        for stamp, relative_pose_transform in poses:
            if self.direction_kalman_filter.initialized:
                previous_yaw = self.direction_kalman_filter.get_yaw()
                curr_yaw = self.direction_kalman_filter.predict(relative_pose_transform)
            
            if self.position_kalman_filters.initialized:
                self.position_kalman_filters.predict(relative_pose_transform, previous_yaw, curr_yaw)

        # clear the queue up to the current pose
        self.relative_transform_queue = self.relative_transform_queue[idx:]

    def get_relative_transform_in_wire_cam(self, relative_pose_transform):
        """
        Get the relative pose transform in the wire camera frame.
        """
        relative_pose_transform = np.array(relative_pose_transform).reshape(4, 4)
        relative_pose_transform_wire_cam = self.H_pose_to_wire @ relative_pose_transform @ self.H_wire_to_pose
        return relative_pose_transform_wire_cam
        
    def wire_detection_callback(self, wire_detections_msg):
        """
        Callback for the wire detections message. This is used to update the Kalman filters with the new wire detections.
        """
        if not self.initialized:
            self.get_logger().info("Camera info not received yet. Waiting for camera info.")
            return
        
        wire_detection_stamp = wire_detections_msg.header.stamp.sec + wire_detections_msg.header.stamp.nanosec * 1e-9
        if len(self.relative_transform_queue) != 0:
            # Update the Kalman filters with the poses up to the current wire detection timestamp
            self.predict_kfs_up_to_current_pose(wire_detection_stamp)
        
        if wire_detections_msg.avg_angle is None or wire_detections_msg.avg_angle == 0.0 or len(wire_detections_msg.wire_detections) == 0:
            return
        
        # Get the current pose in world frame
        wire_points_xyz = np.zeros((len(wire_detections_msg.wire_detections), 3))
        wire_directions = np.zeros((len(wire_detections_msg.wire_detections), 3))
        for i, wire_detection in enumerate(wire_detections_msg.wire_detections):
            start = np.array([wire_detection.start.x, wire_detection.start.y, wire_detection.start.z])
            end = np.array([wire_detection.end.x, wire_detection.end.y, wire_detection.end.z])
            wire_points_xyz[i, :] = (start + end) / 2.0
            if self.direction_kalman_filter.initialized:
                wire_directions[i, :] = self.direction_kalman_filter.get_direction_from_line_end_points(start, end)
            elif not self.direction_kalman_filter.initialized and i == 0:
                reference_dir = self.direction_kalman_filter.get_direction_from_line_end_points(start, end)
                wire_directions[i, :] = reference_dir
            else:
                wire_directions[i, :] = self.direction_kalman_filter.get_direction_from_line_end_points(start, end, reference_dir)

        avg_wire_direction = np.mean(wire_directions, axis=0) / np.linalg.norm(np.mean(wire_directions, axis=0))
        if self.direction_kalman_filter.initialized:
            self.direction_kalman_filter.update(avg_wire_direction)
            self.get_logger().info(f"Average measured yaw: {np.rad2deg(np.arctan2(avg_wire_direction[1], avg_wire_direction[0])):.2f} degrees")
            self.get_logger().info(f"Current wire yaw: {np.rad2deg(self.direction_kalman_filter.get_yaw()):.2f} degrees")
        else:
            self.get_logger().info("Initializing Direction filter with new wire direction.")
            self.direction_kalman_filter.initialize(avg_wire_direction)

        if self.position_kalman_filters.initialized:
            self.position_kalman_filters.update(wire_points_xyz, self.direction_kalman_filter.get_yaw())
        else:
            self.get_logger().info("Initializing Position Filters with new wire locations.")
            self.position_kalman_filters.initialize_kfs(wire_points_xyz, self.direction_kalman_filter.get_yaw())
        self.total_iterations += 1

        if self.wire_viz_2d:
            # find closest rgb timestamp to the wire detection timestamp
            rgb_index = bisect.bisect_right(self.rgb_timestamps, wire_detection_stamp) - 1
            if rgb_index < len(self.rgb_timestamps) and rgb_index >= 0:
                rgb_stamp = self.rgb_timestamps[rgb_index]
                rgb_image = self.rgb_imgs[rgb_index]
                # Draw the Kalman filters on the RGB image
                img = self.draw_kfs_viz(rgb_image, wire_detections=wire_detections_msg.wire_detections)
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header.stamp.sec = int(rgb_stamp)
                img_msg.header.stamp.nanosec = int((rgb_stamp - int(rgb_stamp)) * 1e9)
                self.tracking_2d_pub.publish(img_msg)

        if self.wire_viz_3d:
            # Draw the Kalman filters on the 3D visualization
            self.visualize_3d_wires()
                                            
    def target_timer_callback(self):
        if self.total_iterations < self.iteration_start_threshold or not self.initialized:
            return
        
        if self.tracked_wire_id is None:
            self.tracked_wire_id = self.position_kalman_filters.get_target_id()
            if self.tracked_wire_id is None:
                raise ValueError("No valid wire ID found for tracking.")
        if self.tracked_wire_id not in self.position_kalman_filters.kf_ids:
            raise ValueError("Tracked wire ID not found in Kalman filters.")
        
        if self.tracked_wire_id is not None and self.position_kalman_filters.initialized:
            target_kf = self.position_kalman_filters.get_kf_by_id(self.tracked_wire_id)
            target_theta, target_rho, target_pitch, target_xyz = self.transform_kf_id_to_2d_plucker(self.tracked_wire_id)
            if target_kf is None:
                self.get_logger().info("No valid Kalman filter found for tracked wire ID.")
                return
            
            target_msg = WireTarget()
            target_msg.target_angle = target_theta
            target_msg.target_distance = target_rho
            target_msg.target_pitch = target_pitch
            target_msg.target_image_x = target_xyz[0]
            target_msg.target_image_y = target_xyz[1]
            target_msg.target_height = target_xyz[2]

            self.wire_target_pub.publish(target_msg)

    def debug_kfs(self):
        if len(self.position_kalman_filters) > 0:
            valid_counts = []
            kf_midpoints = []
            for i, kf in self.position_kalman_filters.items():
                kf_midpoints.append(kf.curr_pos)
                valid_counts.append(kf.valid_count)

            self.get_logger().info("Detected Midpoints: %s" % str(kf_midpoints))
            self.get_logger().info("valid_counts: %s" % str(valid_counts))
            self.get_logger().info("Num KFs: %s" % len(self.position_kalman_filters))

        if self.direction_kalman_filter is not None:
            yaw = self.direction_kalman_filter.get_yaw()
            self.get_logger().info("Wire global Yaw: %s" % yaw)

    def draw_kfs_viz(self, img, wire_detections=None):
        # assumes a up, right, forward world frame
        valid_kfs_ids = np.where(self.position_kalman_filters.valid_counts > self.min_valid_kf_count_threshold)[0]
        if len(valid_kfs_ids) != 0:
            valid_colors = self.position_kalman_filters.kf_colors[valid_kfs_ids.flatten(), :]
            wire_direction_estimate = self.direction_kalman_filter.get_direction()
            wire_yaw = self.direction_kalman_filter.get_yaw()

            valid_xyz_points = self.position_kalman_filters.get_kf_xyzs(wire_yaw, inds=valid_kfs_ids)
            start_points = valid_xyz_points + 20 * wire_direction_estimate
            end_points = valid_xyz_points - 20 * wire_direction_estimate
            image_start_points = self.camera_matrix @ start_points.T
            image_end_points = self.camera_matrix @ end_points.T
            target_kf_ind = self.position_kalman_filters.kf_ids == self.tracked_wire_id
            for i, (start, end, color) in enumerate(zip(image_start_points.T, image_end_points.T, valid_colors)):
                start = (int(start[0] / start[2]), int(start[1] / start[2]))
                end = (int(end[0] / end[2]), int(end[1] / end[2]))
                cv2.line(img, start, end, color.tolist(), 2)
                center = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
                cv2.circle(img, center, 5, color.tolist(), -1)
                if i == np.squeeze(target_kf_ind)[0]:
                    cv2.circle(img, center, 10, (0, 255, 0), 2)

        if wire_detections is not None:
            for wire_detection in wire_detections:
                start = (wire_detection.start.x, wire_detection.start.y, wire_detection.start.z)
                end = (wire_detection.end.x, wire_detection.end.y, wire_detection.end.z)
                image_start = self.camera_matrix @ np.array([start[0], start[1], start[2]]).T
                image_end = self.camera_matrix @ np.array([end[0], end[1], end[2]]).T
                start = (int(image_start[0] / image_start[2]), int(image_start[1] / image_start[2]))
                end = (int(image_end[0] / image_end[2]), int(image_end[1] / image_end[2]))
                cv2.line(img, start, end, (255, 0, 0), 2)

        return img
    
    def visualize_3d_wires(self):
        if self.position_kalman_filters and self.initialized:
            # Publish wire visualization
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lines"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            line_scale = 0.05
            marker.scale.x = line_scale
            marker.scale.y = line_scale
            marker.scale.z = line_scale

            kf_ids = self.position_kalman_filters.kf_ids
            kf_xyzs = self.position_kalman_filters.get_kf_xyzs(self.direction_kalman_filter.get_yaw())
            assert kf_xyzs.shape[0] == len(kf_ids), f"Kalman filter IDs and XYZs do not match in length, got {len(kf_ids)} and {kf_xyzs.shape[0]} respectively."
            wire_direction = self.direction_kalman_filter.get_direction()
            start_points = kf_xyzs + 1.0 * wire_direction
            end_points = kf_xyzs - 1.0 * wire_direction
            for i in range(len(kf_ids)):
                start = start_points[i,:].astype(np.float32)
                end = end_points[i,:].astype(np.float32)

                # Create Point objects for the start and end points
                p1 = Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
                p2 = Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
                marker.points.append(p1)
                marker.points.append(p2)

                if kf_ids[i] == self.tracked_wire_id:
                    marker.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
                    marker.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))
                    
                else:
                    marker.colors.append(ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))
                    marker.colors.append(ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0))

            # self.get_logger().info(f"Publishing {len(kf_ids)} wire Kalman filters in 3D visualization.")
            self.tracking_3d_pub.publish(marker)

    def transform_kf_id_to_2d_plucker(self, kf_id):
        kf_dh, kf_index = self.position_kalman_filters.get_kf_by_id(kf_id)
        if kf_dh is None:
            self.get_logger().info(f"No Kalman filter found for ID {kf_id}.")
            return None
        
        wire_direction = self.direction_kalman_filter.get_direction()
        kf_xyz = np.squeeze(self.position_kalman_filters.get_kf_xyzs(self.direction_kalman_filter.get_yaw(), inds=np.array(kf_index)))
        start_point = kf_xyz + 20 * wire_direction
        end_point = kf_xyz - 20 * wire_direction
        assert start_point.shape == (3,), f"Start point shape: {start_point.shape}, expected (3,)"
        assert end_point.shape == (3,), f"End point shape: {end_point.shape}, expected (3,)"
        start_point_2d = (self.camera_matrix @ start_point.reshape(3, 1)).flatten()
        end_point_2d = (self.camera_matrix @ end_point.reshape(3, 1)).flatten()
        kf_point_2d = (self.camera_matrix @ kf_xyz.reshape(3, 1)).flatten()

        start_point_2d = (int(start_point_2d[0] / start_point_2d[2]), int(start_point_2d[1] / start_point_2d[2]))
        end_point_2d = (int(end_point_2d[0] / end_point_2d[2]), int(end_point_2d[1] / end_point_2d[2]))
        kf_point_2d = (int(kf_point_2d[0] / kf_point_2d[2]), int(kf_point_2d[1] / kf_point_2d[2]), kf_dh[1])

        # pitch is z change per pixel change in 2D
        pitch = np.arctan2(start_point[2] - end_point[2], np.linalg.norm(np.array(start_point_2d) - np.array(end_point_2d)))
        # theta is in the image plane, from end to start
        theta = np.arctan2(start_point_2d[1] - end_point_2d[1], start_point_2d[0] - end_point_2d[0])
        rho = np.linalg.norm((np.array(start_point_2d) + np.array(end_point_2d)) / 2.0)
        return theta, rho, pitch, kf_point_2d
    
    def rgb_callback(self, rgb_msg):
        """
        Callback for the RGB image. This is used to visualize the wire tracking.
        """
        if not self.initialized:
            return
        
        if len(self.rgb_timestamps) >= 10:
            self.rgb_timestamps.pop(0)
            self.rgb_imgs.pop(0)
        stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        bisect.insort(self.rgb_timestamps, stamp)
        index = self.rgb_timestamps.index(stamp)
        self.rgb_imgs.insert(index, rgb)
        
    def set_params(self):
        self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
        self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
        self.declare_parameter('pose_sub_topic', rclpy.Parameter.Type.STRING)
        self.pose_sub_topic = self.get_parameter('pose_sub_topic').get_parameter_value().string_value
        self.declare_parameter('wire_detections_pub_topic', rclpy.Parameter.Type.STRING)
        self.wire_detections_pub_topic = self.get_parameter('wire_detections_pub_topic').get_parameter_value().string_value
    
        # wire pub topics
        self.declare_parameter('wire_target_pub_topic', rclpy.Parameter.Type.STRING)
        self.wire_target_pub_topic = self.get_parameter('wire_target_pub_topic').get_parameter_value().string_value

        # visulaization pub topics
        self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
        self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value

        self.declare_parameter('tracking_2d_pub_topic', rclpy.Parameter.Type.STRING)
        self.tracking_2d_pub_topic = self.get_parameter('tracking_2d_pub_topic').get_parameter_value().string_value
        self.declare_parameter('tracking_3d_pub_topic', rclpy.Parameter.Type.STRING)
        self.tracking_3d_pub_topic = self.get_parameter('tracking_3d_pub_topic').get_parameter_value().string_value

        wire_viz = bool(os.getenv('WIRE_VIZ', None).lower())
        wire_mode = int(os.getenv('WIRE_MODE', None).lower())
        self.wire_viz_2d = wire_viz and wire_mode == 2
        self.wire_viz_3d = wire_viz 

        # KF parameters
        with open(get_package_share_directory('wire_tracking') + '/config/wire_tracking_config.yaml', 'r') as file:
            self.wire_tracking_config = yaml.safe_load(file)

        self.direction_kalman_filter = DirectionKalmanFilter(self.wire_tracking_config)
        self.target_publish_frequency_hz = self.wire_tracking_config['target_publish_frequency_hz']
        self.min_valid_kf_count_threshold = self.wire_tracking_config['min_valid_kf_count_threshold']
        self.iteration_start_threshold = self.wire_tracking_config['iteration_start_threshold']
    

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