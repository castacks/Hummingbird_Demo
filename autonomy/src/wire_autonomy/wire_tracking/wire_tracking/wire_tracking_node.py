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
from rclpy.callback_groups import ReentrantCallbackGroup
# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ament_index_python.packages import get_package_share_directory

from wire_interfaces.msg import WireDetections

import wire_detection.wire_detection_utils as wdu
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
        self.yaw_kalman_filter = None

        self.previous_pose = None

        self.max_kf_label = 0
        self.total_iterations = 0

        self.tracked_wire_id = None

        # Transform from pose_cam to wire_cam
        # 180 rotation about y-axis, 0.216m translation in negative x-axis
        self.H_wire_cam_to_pose_cam = np.array([[np.cos(np.deg2rad(180)),  0.0, np.sin(np.deg2rad(180)), - 0.216],
                                              [         0.0,             1.0,           0.0,           0.0],
                                              [-np.sin(np.deg2rad(180)), 0.0, np.cos(np.deg2rad(180)), 0.0],
                                              [0.0, 0.0, 0.0, 1.0]])        

        # Subscribers
        self.initialized = False
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1)

        self.pose_sub = self.create_subscription(Odometry, self.pose_sub_topic, self.pose_callback, rclpy.qos.qos_profile_sensor_data)
        self.wire_detection_sub = self.create_subscription(WireDetections, self.wire_detections_pub_topic, self.wire_detection_callback, 1)

        if self.vizualize_wires:
            self.rgb_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.rgb_callback, rclpy.qos.qos_profile_sensor_data)


        # Wire Publishers
        self.wire_target_pub = self.create_publisher(PoseStamped, self.wire_target_pub_topic, 10)

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
        self.previous_downward_pose = pose_msg.pose

        if self.yaw_kalman_filter is not None:
            self.yaw_kalman_filter.predict(relative_pose_transform)

    def get_relative_transform_in_wire_cam(self, relative_pose_transform):
        wire_translations = np.array([[relative_pose_transform[3, 1], relative_pose_transform[3, 0], -relative_pose_transform[3, 2]]]).T
                                                                

    def target_timer_callback(self):

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

        if self.yaw_kalman_filter is not None:
            yaw = self.yaw_kalman_filter.get_yaw()
            self.get_logger().info("Wire global Yaw: %s" % yaw)

    def draw_kfs_viz(self, img, wire_detections, pose_in_world):
        green = (0, 255, 0)
        for i in range(len(wire_lines)):
            
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
            self.declare_parameter('wire_detections_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_detections_pub_topic = self.get_parameter('wire_detections_pub_topic').get_parameter_value().string_value
        
            # wire pub topics
            self.declare_parameter('wire_target_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_target_pub_topic = self.get_parameter('wire_target_pub_topic').get_parameter

            # visulaization pub topics
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.declare_parameter('wire_2d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_2d_viz_pub_topic = self.get_parameter('wire_2d_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_pc_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_pc_pub_topic = self.get_parameter('depth_pc_pub_topic').get_parameter_value().string_value
            self.declare_parameter('wire_3d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_3d_viz_pub_topic = self.get_parameter('wire_3d_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('vizualize_wires', rclpy.Parameter.Type.BOOL)
            self.vizualize_wires = self.get_parameter('vizualize_wires').get_parameter_value().bool_value

            # KF parameters
            with open(get_package_share_directory('wire_tracking') + '/config/wire_tracking_config.yaml', 'r') as file:
                self.wire_tracking_config = yaml.safe_load(file)

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