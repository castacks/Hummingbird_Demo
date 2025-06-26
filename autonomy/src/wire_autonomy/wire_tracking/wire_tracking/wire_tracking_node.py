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
import yaml

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ament_index_python.packages import get_package_share_directory

from wire_interfaces.msg import WireDetections

import wire_detection.wire_detection_utils as wdu
import wire_tracking.coord_transforms as ct
from wire_tracking.kalman_filters import PositionKalmanFilters, DirectionKalmanFilter

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ZED_BASELINE_M = .12 / 2

class WireTrackingNode(Node):
    def __init__(self):
        super().__init__('wire_tracking_node')
        self.set_params()

        self.position_kalman_filters = {}
        self.direction_kalman_filter = None

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
        self.H_pose_to_wire = np.zeros((4, 4))
        self.H_pose_to_wire[:3, :3] = self.Rz @ self.Ry        
        self.H_pose_to_wire[3, :3] = np.array([ZED_BASELINE_M, 0.0, -0.216]).T
        self.H_wire_to_pose = np.linalg.inv(self.H_pose_to_wire)

        # Subscribers
        self.initialized = False
        camera_info_callback_group = MutuallyExclusiveCallbackGroup()
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1, callback_group=camera_info_callback_group)

        pose_callback_group = ReentrantCallbackGroup()
        self.pose_sub = self.create_subscription(Odometry, self.pose_sub_topic, self.pose_callback, rclpy.qos.qos_profile_sensor_data, callback_group=pose_callback_group)
        self.wire_detection_sub = self.create_subscription(WireDetections, self.wire_detections_pub_topic, self.wire_detection_callback, 1, callback_group=pose_callback_group)

        if self.vizualize_wires:
            self.rgb_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.rgb_callback, rclpy.qos.qos_profile_sensor_data)

        # Wire Publishers
        self.wire_target_pub = self.create_publisher(PoseStamped, self.wire_target_pub_topic, 10)
        self.wire_target_pub_timer = self.create_timer(1.0 / self.target_publish_frequency_hz, self.target_timer_callback)

        # Visualization Publishers
        self.tracking_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 10)
        self.pose_viz_pub = self.create_publisher(PoseStamped, self.pose_viz_pub_topic, 10)

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
        self.initialized = True
        self.image_shape = (data.height, data.width)
        self.position_kalman_filters = PositionKalmanFilters(self.wire_tracking_config, self.camera_matrix, self.image_shape)
        self.get_logger().info("Wire Tracking Node initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)

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
        self.previous_downward_pose = pose_msg.pose

        if self.direction_kalman_filter.initialized:
            previous_yaw = self.direction_kalman_filter.get_yaw()
            curr_yaw = self.direction_kalman_filter.predict(relative_pose_transform)
        if self.position_kalman_filters.initialized:
            self.position_kalman_filters.predict(relative_wire_transform, previous_yaw, curr_yaw)

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
        if wire_detections_msg.avg_angle is None or wire_detections_msg.avg_angle == 0.0 or len(wire_detections_msg.wire_lines) == 0:
            self.get_logger().info("No wire detections received.")
            return
        
        # Get the current pose in world frame
        wire_points_xyz = np.zeros((len(wire_detections_msg.wire_lines), 3))
        wire_directions = np.zeros((len(wire_detections_msg.wire_lines), 3))
        for i, wire_detection in enumerate(wire_detections_msg.wire_detections):
            start = np.array([wire_detection.start.x, wire_detection.start.y, wire_detection.start.z])
            end = np.array([wire_detection.end.x, wire_detection.end.y, wire_detection.end.z])
            wire_points_xyz[i, :] = (start + end) / 2.0
            if self.direction_kalman_filter.initialized:
                wire_directions[i, :] = self.direction_kalman_filter.get_direction_from_line_end_points(start, end)
            elif not self.direction_kalman_filter.initialized and i == 0:
                reference_dir = self.get_direction_from_line_end_points(start, end)
                wire_directions[i, :] = reference_dir
            else:
                wire_directions[i, :] = self.get_direction_from_line_end_points(start, end, reference_dir)

        avg_wire_direction = np.mean(wire_directions, axis=0) / np.linalg.norm(np.mean(wire_directions, axis=0))
        if self.direction_kalman_filter.initialized:
            self.direction_kalman_filter.update(avg_wire_direction)
        else:
            self.direction_kalman_filter.initialize(avg_wire_direction)

        if self.position_kalman_filters.initialized:
            self.direction_kalman_filter.update(wire_points_xyz, self.direction_kalman_filter.get_yaw())
        else:
            self.direction_kalman_filter.initialize(wire_points_xyz, self.direction_kalman_filter.get_yaw())

        self.total_iterations += 1
                                                                
    def target_timer_callback(self):

        if self.total_iterations > self.iteration_start_threshold:
            if self.tracked_wire_id is None:
                self.tracked_wire_id = self.position_kalman_filters.get_closest_wire()
                if self.tracked_wire_id is None:
                    self.get_logger().info("No wire detected, skipping target publish.")
                    return

        self.publish_wire_target()

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

    def draw_kfs_viz(self, img):
        # assumes a up, right, forward world frame
        valid_kfs_ids = self.position_kalman_filters.valid_counts > self.min_valid_kf_count_threshold
        valid_points = self.position_kalman_filters.kf_points[valid_kfs_ids, :]
        valid_colors = self.position_kalman_filters.kf_colors[valid_kfs_ids, :]
        wire_direction_estimate = self.direction_kalman_filter.get_direction()
        wire_yaw = self.direction_kalman_filter.get_yaw()
        valid_xyz_points = self.position_kalman_filters.get_xys_from_dists(valid_points[:, 0], wire_direction_estimate)
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
            if i == target_kf_ind:
                cv2.circle(img, center, 10, (0, 255, 0), 2)
        return img
    
    def rgb_callback(self, rgb_msg):
        """
        Callback for the RGB image. This is used to visualize the wire tracking.
        """
        if not self.initialized:
            return
        
        bridge = CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        
        if self.position_kalman_filters.initialized:
            rgb = self.draw_kfs_viz(rgb)
        
        viz_pub_msg = bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        self.tracking_viz_pub.publish(viz_pub_msg)
        
    def set_params(self):
        try:
            # sub topics
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.declare_parameter('pose_sub_topic', rclpy.Parameter.Type.STRING)
            self.pose_sub_topic = self.get_parameter('pose_sub_topic').get_parameter_value().string_value
            self.declare_parameter('wire_detections_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_detections_pub_topic = self.get_parameter('wire_detections_pub_topic').get_parameter_value().string_value
        
            # wire pub topics
            self.declare_parameter('wire_target_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_target_pub_topic = self.get_parameter('wire_target_pub_topic').get_parameter

            # visulaization pub topics
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.declare_parameter('tracking_2d_pub_topic', rclpy.Parameter.Type.STRING)
            self.tracking_2d_pub_topic = self.get_parameter('tracking_2d_pub_topic').get_parameter_value().string_value
            self.declare_parameter('tracking_3d_pub_topic', rclpy.Parameter.Type.STRING)
            self.tracking_3d_pub_topic = self.get_parameter('tracking_3d_pub_topic').get_parameter_value().string_value
            self.declare_parameter('vizualize_wires', rclpy.Parameter.Type.BOOL)
            self.vizualize_wires = self.get_parameter('vizualize_wires').get_parameter_value().bool_value

            # KF parameters
            with open(get_package_share_directory('wire_tracking') + '/config/wire_tracking_config.yaml', 'r') as file:
                self.wire_tracking_config = yaml.safe_load(file)

            self.direction_kalman_filter = DirectionKalmanFilter(self.wire_tracking_config)
            self.target_publish_frequency_hz = self.wire_tracking_config['target_publish_frequency_hz']
            self.min_valid_kf_count_threshold = self.wire_tracking_config['min_valid_kf_count_threshold']
            self.iteration_start_threshold = self.wire_tracking_config['iteration_start_threshold']

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