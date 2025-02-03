#!/usr/bin/env python

import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from common_utils.wire_detection import WireDetector, find_closest_point_on_3d_line, clamp_angles_pi, create_depth_viz
from common_utils.kalman_filters import PositionKalmanFilter, YawKalmanFilter
import common_utils.coord_transforms as ct

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireTrackingNode(Node):
    def __init__(self):
        super().__init__('wire_tracking_node')
        self.set_params()

        # Wire Detector
        self.wire_detector = WireDetector(threshold=self.line_threshold, expansion_size=self.expansion_size)
        self.position_kalman_filters = {}
        self.vis_colors = {}
        self.yaw_kalman_filter = None
        self.are_kfs_initialized = False
        self.max_kf_label = 0
        self.line_length = None
        self.tracked_wire_id = None
        self.total_iterations = 0

        # Transform from pose_cam to wire_cam
        # 180 rotation about x-axis, 0.216m translation in y-axis
        self.pose_cam_to_wire_cam = np.array([[1.0, 0.0, 0.0, 0.0],
                                             [0.0, np.cos(np.deg2rad(180)), -np.sin(np.deg2rad(180)), 0.216],
                                             [0.0, np.sin(np.deg2rad(180)), np.cos(np.deg2rad(180)), 0.0],
                                             [0.0, 0.0, 0.0, 1.0]])

        # Subscribers
        cam_info_callbackgroup = ReentrantCallbackGroup()
        self.received_camera_info = False
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 
                                                        rclpy.qos.qos_profile_sensor_data, 
                                                        callback_group=cam_info_callbackgroup)
        self.rgb_image_sub = Subscriber(self, Image, self.rgb_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.depth_image_sub = Subscriber(self, Image, self.depth_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.pose_sub = Subscriber(self, PoseStamped, self.pose_sub_topic)
        self.bridge = CvBridge()

        # Time Synchronizer
        self.img_tss = ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub, self.pose_sub],
            queue_size=1, 
            slop=0.2
        )
        self.img_tss.registerCallback(self.input_callback)

        # Publishers
        self.tracking_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 10)
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 10)
        self.pose_viz_pub = self.create_publisher(PoseStamped, self.pose_viz_pub_topic, 10)

        self.get_logger().info("Wire Tracking Node initialized")
        
    def camera_info_callback(self, data):
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.camera_vector = np.array([self.fx, self.fy, self.cx, self.cy])
        if self.received_camera_info == False:
            self.get_logger().info("Received Camera Info")
        self.received_camera_info = True

    def input_callback(self, rgb_msg, depth_msg, pose_msg):
        if self.line_length is None:
            self.line_length = max(rgb_msg.width, rgb_msg.height) * 2
        try:
            # Convert the ROS image messages to OpenCV images
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        
        debug_image = None
        if self.received_camera_info:
            # transform pose cam pose to wire cam pose
            pose = self.transform_pose_cam_to_wire_cam(pose_msg.pose)
            # self.get_logger().info(f"Wire cam Pose: {pose}")
            # self.get_logger().info(f"Pose cam Pose: {pose_msg.pose}")
            debug_image = self.detect_lines_and_update(rgb, depth, pose)

        if debug_image is None:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(rgb, "rgb8")
        else:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(debug_image, "rgb8")
        self.tracking_viz_pub.publish(viz_pub_msg) 

        self.debug_kfs()

        # create depth visualization
        depth_viz = create_depth_viz(depth)
        depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, "rgb8")
        self.depth_viz_pub.publish(depth_viz_msg)

        # publish the wire cam pose
        new_pose_msg = PoseStamped()
        new_pose_msg.header = pose_msg.header
        new_pose_msg.pose = pose
        self.pose_viz_pub.publish(new_pose_msg)

    def detect_lines_and_update(self, image, depth, pose: Pose):
        # get a segmentation mask from the rgb image
        seg_mask = self.wire_detector.create_seg_mask(image)

        # if there are no lines detected, return None, a default image will be published
        debug_img = None
        if np.any(seg_mask):
            wire_lines, wire_midpoints, avg_yaw = self.wire_detector.detect_wires(seg_mask)
            # get the horizontal camera yaw
            cam_yaw = ct.get_yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            if len(wire_midpoints) != 0:
                global_yaw = cam_yaw + avg_yaw
                global_yaw = clamp_angles_pi(global_yaw) 
                corresponding_depths = np.zeros(len(wire_midpoints))
                for i, (x, y) in enumerate(wire_midpoints):
                    # does not allow for the depth of a midpoint to be a indescipt value
                    depth_midpoint = depth[y, x]
                    # TODO: refine this method
                    while depth_midpoint == 0.0 or depth_midpoint == None or np.isnan(depth_midpoint):
                        x = int(x + np.cos(avg_yaw)*2)
                        y = int(y + np.sin(avg_yaw)*2) 
                        depth_midpoint = depth[y, x]

                    corresponding_depths[i] = depth_midpoint

                global_midpoints = ct.image_to_world_pose(wire_midpoints, corresponding_depths, pose, self.camera_vector)
                if self.are_kfs_initialized == False:
                    self.initialize_kfs(global_midpoints, global_yaw)
                else:
                    self.update_kfs(global_midpoints, global_yaw, pose)

                # self.debug_kfs(pose=pose)

            self.total_iterations += 1
            if self.tracked_wire_id is None and self.total_iterations > self.target_start_threshold:
                min_distance = 100000
                min_id = None
                for i, kf in self.position_kalman_filters.items():
                    if kf.valid_count >= self.valid_threshold:
                        curr_pos = pose.position
                        distance = ct.get_distance_between_3D_point(kf.curr_pos, np.array([curr_pos.x, curr_pos.y, curr_pos.z]))
                        if distance < min_distance:
                            min_distance = distance
                            min_id = i
                self.tracked_wire_id = min_id

            if self.are_kfs_initialized:
                debug_img = self.draw_valid_kfs(image, pose)        
        return debug_img
    
    def transform_pose_cam_to_wire_cam(self, pose_cam_pose):
        world_to_pose_cam = ct.pose_to_homogeneous(pose_cam_pose)
        pose_wire_transform = world_to_pose_cam @ self.pose_cam_to_wire_cam
        wire_cam_pose = ct.homogeneous_to_pose(pose_wire_transform)
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

            self.yaw_kalman_filter = YawKalmanFilter(global_yaw)
            self.are_kfs_initialized = True
            self.get_logger().info("Kalman Filters Initialized")

    def add_kf(self, midpoint):
        self.max_kf_label += 1
        self.position_kalman_filters[self.max_kf_label] = PositionKalmanFilter(midpoint)
        self.position_kalman_filters[self.max_kf_label].valid_count = 1
        saturation_threshold = 150
        value_threshold = 100
        value = 0
        saturation = 0
        while saturation < saturation_threshold and value < value_threshold:
            color = np.random.randint(0, 256, 3).tolist()
            color_np = np.array(color).reshape(1, 1, 3).astype(np.uint8)
            hsv = cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)
            saturation = hsv[0, 0, 1]
            value = hsv[0, 0, 2]
        self.vis_colors[self.max_kf_label] = color

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
                closest_point = find_closest_point_on_3d_line(midpoint, self.yaw_kalman_filter.get_yaw(), kf.curr_pos)
                distance = np.linalg.norm(closest_point - kf.curr_pos)
                # if the closest point on the detected line is close enough to an existing Kalman Filter, update it
                if distance < self.distance_threshold:
                    kf.update(midpoint)
                    matched_kf_ids.append(kf_id)
                    got_matched = True
                    break
            if not got_matched:
                new_midpoints.append(midpoint)

        # Add new Kalman Filters for unmatched midpoints
        # self.get_logger().info(f'adding {len(new_midpoint)} new Kalman Filters')
        for midpoint in new_midpoints:
            self.add_kf(midpoint)

        # Remove Kalman Filters that have not been updated for a while
        kfs_to_remove = []
        for kf_id, kf in self.position_kalman_filters.items():
            if kf_id not in matched_kf_ids:
                x_img, y_img = ct.world_to_image_pose(kf.curr_pos[0], kf.curr_pos[1], kf.curr_pos[2], pose, self.camera_vector)
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

    def draw_valid_kfs(self, img, pose_in_world):
        global_yaw = self.yaw_kalman_filter.get_yaw()
        cam_yaw = ct.get_yaw_from_quaternion(pose_in_world.orientation.x, pose_in_world.orientation.y, pose_in_world.orientation.z, pose_in_world.orientation.w)
        image_yaw = global_yaw - cam_yaw
        for i, kf in self.position_kalman_filters.items():
            if kf.valid_count >= self.valid_threshold:
                image_midpoint = ct.world_to_image_pose(kf.curr_pos[0], kf.curr_pos[1], kf.curr_pos[2], pose_in_world, self.camera_vector)
                x0 = int(image_midpoint[0] + self.line_length * np.cos(image_yaw))
                y0 = int(image_midpoint[1] + self.line_length * np.sin(image_yaw))
                x1 = int(image_midpoint[0] - self.line_length * np.cos(image_yaw))
                y1 = int(image_midpoint[1] - self.line_length * np.sin(image_yaw))
                cv2.line(img, (x0, y0), (x1, y1), self.vis_colors[i], 2)
                cv2.circle(img, (int(image_midpoint[0]), int(image_midpoint[1])), 5, self.vis_colors[i], -1)

        if self.tracked_wire_id is not None:
            tracked_midpoint = self.position_kalman_filters[self.tracked_wire_id].curr_pos
            image_midpoint = ct.world_to_image_pose(tracked_midpoint[0], tracked_midpoint[1], tracked_midpoint[2], pose_in_world, self.camera_vector)
            cv2.ellipse(img, (int(image_midpoint[0]), int(image_midpoint[1])), (15, 15), 0, 0, 360, (0, 255, 0), 3)
            
        return img
        
    def set_params(self):
        try:
            # Subscriber topics
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('pose_sub_topic', rclpy.Parameter.Type.STRING)

            # Publisher topics
            self.declare_parameter('wire_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('pose_viz_pub_topic', rclpy.Parameter.Type.STRING)

            # Wire Detection parameters
            self.declare_parameter('line_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('expansion_size', rclpy.Parameter.Type.INTEGER)

            # KF parameters
            self.declare_parameter('max_distance_threshold', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('min_valid_kf_count_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('iteration_start_threshold', rclpy.Parameter.Type.INTEGER)

            # Access parameters
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value
            self.pose_sub_topic = self.get_parameter('pose_sub_topic').get_parameter_value().string_value

            self.wire_viz_pub_topic = self.get_parameter('wire_viz_pub_topic').get_parameter_value().string_value
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.pose_viz_pub_topic = self.get_parameter('pose_viz_pub_topic').get_parameter_value().string_value

            self.line_threshold = self.get_parameter('line_threshold').get_parameter_value().integer_value
            self.expansion_size = self.get_parameter('expansion_size').get_parameter_value().integer_value

            self.distance_threshold = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
            self.valid_threshold = self.get_parameter('min_valid_kf_count_threshold').get_parameter_value().integer_value
            self.target_start_threshold = self.get_parameter('iteration_start_threshold').get_parameter_value().integer_value

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