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

    def local_to_global_yaw(self, image_yaw, H_wire_cam_pose):
        """
        Convert the local image yaw to a global yaw based on the pose of the camera in the world frame.
        """
        # Convert image yaw to radians
       
        
        # Get the yaw from the pose in wire cam frame
        cam_yaw_vector = np.array([np.cos(image_yaw), np.sin(image_yaw), 0.0]).reshape(-1, 1)
        global_yaw_vector = H_wire_cam_pose[:3, :3] @ cam_yaw_vector
        global_yaw = np.arctan2(global_yaw_vector[1], global_yaw_vector[0])
        global_yaw = wdu.fold_angles_from_0_to_pi(global_yaw)
        return global_yaw
    
    def pose_cam_pose_to_H_wire_cam(self, pose_cam_pose):
        """
        Convert the pose in the camera frame to a homogeneous transformation matrix in the wire cam frame.
        """
        # Convert the pose to a homogeneous transformation matrix
        H_pose_cam_to_world = ct.pose_to_homogeneous(pose_cam_pose)
        # Transform to wire cam frame
        H_wire_cam_to_world = self.H_wire_cam_to_pose_cam @ H_pose_cam_to_world
        return H_wire_cam_to_world
        
    def pose_detection_callback(self, pose_msg):

        if self.previous_pose is None:
            self.previous_pose = pose_msg.pose
            return
        
        relative_pose = ct.relative_pose_transform(self.previous_pose, pose_msg.pose)


    def pose_detection_callback(self, wire_detection_msg, pose_msg):
        if not self.initialized:
            return
        
        # deal with the wire angle if its in the detection message
        if wire_detection_msg.avg_angle:
            H_wire_cam_to_world = self.pose_cam_pose_to_H_wire_cam(pose_msg.pose)
            global_yaw = self.local_to_global_yaw(wire_detection_msg.avg_angle, H_wire_cam_to_world)
            if self.yaw_kalman_filter == None:
                self.yaw_kalman_filter = YawKalmanFilter(global_yaw, self.wire_tracking_config)
            else:
                self.yaw_kalman_filter.update(global_yaw)
        else:
            if self.yaw_kalman_filter != None:
                self.yaw_kalman_filter.predict()

        if len(wire_detection_msg.wire_detections) > 0 and wire_detection_msg.avg_angle:
            self.wire_detection_points = wire_detection_msg.wire_detections
            midpoints_in_cam = np.array([
                [wire_detection.midpoint.x, wire_detection.midpoint.y, wire_detection.midpoint.z]
                for wire_detection in wire_detection_msg.wire_detections
            ])
            # Transform midpoints to world frame
            yz_in_world = ct.points_in_cam_to_world(midpoints_in_cam, H_wire_cam_to_world)[:, 1:3]  # Get only y and z coordinates
            if len(self.position_kalman_filters) == 0 or self.position_kalman_filters == {}:
                for midpoint in yz_in_world:
                    self.add_kf(midpoint)
            else:
                self.update_pos_kfs(yz_in_world)
        else:
            for kf in self.position_kalman_filters.values:
                kf.predict()

        self.total_iterations += 1

        # self.debug_kfs()

        if self.tracked_wire_id is None and self.total_iterations > self.target_start_threshold:
            min_depth = np.inf
            min_id = None
            for i, kf in self.position_kalman_filters.items():
                if kf.valid_count >= self.min_valid_kf_count_threshold:
                    depth = kf.curr_pos[2]
                    if depth < min_depth:
                        min_depth = depth
                        min_id = i
            self.tracked_wire_id = min_id     

        self.publish_wire_target()

    def pose_detection_rgb_callback(self, wire_detection_msg, pose_msg, rgb_msg):
        if not self.initialized:
            return
        self.pose_detection_callback(wire_detection_msg, pose_msg)

        self.draw_kfs_viz


    def add_kf(self, midpoint):
        self.max_kf_label += 1
        y0, z0 = midpoint[1], midpoint[2]
        self.position_kalman_filters[self.max_kf_label] = PositionKalmanFilter(y0, z0, self.wire_tracking_config)
        return self.max_kf_label

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

    def update_pos_kfs(self, new_global_yz_midpoints):
        assert len(new_global_yz_midpoints.shape) == 2, f"new_global_yz_midpoints should be 2D wtih a y and z value, got {new_global_yz_midpoints.shape}"
        matched_kf_ids = []
        new_midpoints = []
        for midpoint in new_global_yz_midpoints:
            got_matched = False
            for kf_id, kf in self.position_kalman_filters.items():
                distance = np.linalg.norm(kf.curr_pos - midpoint.T)
                # if the closest point on the detected line is close enough to an existing Kalman Filter, update it
                if distance < self.wire_matching_min_threshold_m:
                    kf.update(midpoint)
                    matched_kf_ids.append(kf_id)
                    got_matched = True
                    break
            if not got_matched:
                new_midpoints.append(midpoint)

        # Add new Kalman Filters for unmatched midpoints
        for midpoint in new_midpoints:
            matched_kf_ids.append(self.add_kf(midpoint))

        # Remove Kalman Filters that have not been updated for a while
        for kf_id in list(self.position_kalman_filters.keys()):
            if kf_id not in matched_kf_ids:
                kf = self.position_kalman_filters[kf_id]
                kf.valid_count -= 1

                if kf.valid_count < 0:
                    if kf_id == self.tracked_wire_id:
                        self.tracked_wire_id = None
                        self.get_logger().info("Tracked wire was removed")
                    self.position_kalman_filters.pop(kf_id)

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

            self.wire_matching_min_threshold_m = self.wire_tracking_config['wire_matching_min_threshold_m']
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