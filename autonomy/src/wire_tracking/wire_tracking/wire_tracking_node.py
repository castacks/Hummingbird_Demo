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

import common_utils.wire_detection as wd
import common_utils.coord_transforms as ct
from common_utils.kalman_filters import PositionKalmanFilter, YawKalmanFilter

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireTrackingNode(Node):
    def __init__(self):
        super().__init__('wire_tracking_node')
        self.set_params()

        # Wire Detector
        self.wire_detector = wd.WireDetector(threshold=self.line_threshold, expansion_size=self.expansion_size)
        self.position_kalman_filters = {}
        self.vis_colors = {}
        self.yaw_kalman_filter = None
        self.are_kfs_initialized = False
        self.max_kf_label = 0
        self.line_length = None
        self.tracked_wire_id = None
        self.total_iterations = 0

        # Transform from pose_cam to wire_cam
        # 180 rotation about y-axis, 0.216m translation in negative z-axis
        self.pose_cam_to_wire_cam = np.array([[np.cos(np.deg2rad(180)),  0.0, np.sin(np.deg2rad(180)), 0.0],
                                              [         0.0,             1.0,           0.0,           0.0],
                                              [-np.sin(np.deg2rad(180)), 0.0, np.cos(np.deg2rad(180)), - 0.216],
                                              [0.0, 0.0, 0.0, 1.0]])

        # Subscribers
        cam_info_callbackgroup = ReentrantCallbackGroup()
        self.received_camera_info = False
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 
                                                        rclpy.qos.qos_profile_sensor_data, 
                                                        callback_group=cam_info_callbackgroup)
        

        self.rgb_image_sub = Subscriber(self, Image, self.rgb_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.depth_image_sub = Subscriber(self, Image, self.depth_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)

        # switch the pose topic based on the use_pose_cam parameter
        if self.use_pose_cam:
            self.pose_sub_topic = '/pose_cam' + self.pose_sub_topic
        else:
            self.pose_sub_topic = '/wire_cam' + self.pose_sub_topic

        self.get_logger().info(f"Using Pose on topic {self.pose_sub_topic}")
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
        
        # pose comes in camera frame orientation, x down, y left, z forward
        if self.use_pose_cam:
            pose = self.transform_pose_cam_to_wire_cam(pose_msg.pose)
        else:    
            pose = pose_msg.pose

        # self.get_logger().info(f"Wire Cam Position: {pose.position.x}, {pose.position.y}, {pose.position.z}")
        # roll, pitch, yaw = ct.quaternion_to_euler(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w, 'xyz')
        # self.get_logger().info(f"X Rot: {np.rad2deg(roll)}, Y Rot: {np.rad2deg(pitch)}, Z Rot: {np.rad2deg(yaw)}")
        # self.get_logger().info(f"Wire Cam Yaw: {np.rad2deg(ct.get_z_rot_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))}")

        debug_image = None
        if self.received_camera_info:
            # transform pose cam pose to wire cam pose
            debug_image = self.detect_lines_and_update(rgb, depth, pose)

        if debug_image is None:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(rgb, "rgb8")
        else:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(debug_image, "rgb8")
        self.tracking_viz_pub.publish(viz_pub_msg) 

        # create depth visualization
        depth_viz = wd.create_depth_viz(depth)
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
        self.get_logger().info(f"hello")
        if np.any(seg_mask):
            wire_lines, wire_midpoints, wire_yaw_in_image = self.wire_detector.detect_wires(seg_mask)
            self.get_logger().info(f"Num wires detected: {len(wire_midpoints)}")

            # get the horizontal camera yaw
            # pose_yaw = ct.get_x_rot_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            if len(wire_midpoints) != 0:
                # global_yaw = wd.clamp_angles_pi(pose_yaw + wire_yaw_in_image)

                global_yaw = 0.0
                corresponding_depths = np.zeros(len(wire_midpoints))
                for i, (x, y) in enumerate(wire_midpoints):

                    # depth_midpoint = depth[y, x]
                    # find all depths that are in the segmentation mask and lie on the line of the wire, and average their depths
                    pt1 = np.array([wire_lines[i][0], wire_lines[i][1]])
                    pt2 = np.array([wire_lines[i][2], wire_lines[i][3]])
                    wire_depth, wire_global_yaw = self.characterize_a_line(pt1, pt2, depth, seg_mask)
                    wire_global_yaw = wd.clamp_angles_pi(wire_global_yaw)
                    global_yaw += wire_global_yaw
                    corresponding_depths[i] = wire_depth

                global_midpoints = ct.image_to_world_pose(wire_midpoints, corresponding_depths, pose, self.camera_vector)
                global_yaw /= len(wire_midpoints)

                if self.are_kfs_initialized == False:
                    self.initialize_kfs(global_midpoints, global_yaw)
                else:
                    self.update_kfs(global_midpoints, global_yaw, pose)

                pose_yaw = ct.clamp_angles_pi(global_yaw - wire_yaw_in_image)
                self.get_logger().info(f"Global Yaw: {global_yaw}, Cam Yaw: {pose_yaw}, Yaw in Image {wire_yaw_in_image}")
                self.debug_kfs(detected_midpoints=global_midpoints, depth_midpoints=corresponding_depths, yaw=global_yaw)

            self.total_iterations += 1
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
                debug_img = self.draw_kfs_viz(image, pose)
            else:
                debug_img = None        
        return debug_img
    
    def transform_pose_cam_to_wire_cam(self, pose_cam_pose):
        world_to_pose_cam = ct.pose_to_homogeneous(pose_cam_pose)
        world_to_wire_cam = world_to_pose_cam @ self.pose_cam_to_wire_cam
        wire_cam_pose = ct.homogeneous_to_pose(world_to_wire_cam)
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
                closest_point = wd.find_closest_point_on_3d_line(midpoint, self.yaw_kalman_filter.get_yaw(), kf.curr_pos)
                distance = np.linalg.norm(closest_point - kf.curr_pos)
                self.get_logger().info(f"Distance: {distance}")
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

    def draw_kfs_viz(self, img, pose_in_world):
        kf_yaw = self.yaw_kalman_filter.get_yaw()
        for i, kf in self.position_kalman_filters.items():
            if kf.valid_count >= self.valid_threshold:
                x0 = kf.curr_pos[0] + 1.0 * np.cos(kf_yaw)
                y0 = kf.curr_pos[1] + 1.0 * np.sin(kf_yaw)
                x1 = kf.curr_pos[0] - 1.0 * np.cos(kf_yaw)
                y1 = kf.curr_pos[1] - 1.0 * np.sin(kf_yaw)

                line_points = ct.world_to_image_pose(np.array([[x0, y0, kf.curr_pos[2]], [x1, y1, kf.curr_pos[2]]]), pose_in_world, self.camera_vector)
                x0, y0 = line_points[0] * self.line_length
                x1, y1 = line_points[1] * self.line_length

                # image_midpoint = ct.world_to_image_pose(kf.curr_pos, pose_in_world, self.camera_vector)
                # x0 = int(image_midpoint[0] + self.line_length * np.cos(image_yaw))
                # y0 = int(image_midpoint[1] + self.line_length * np.sin(image_yaw))
                # x1 = int(image_midpoint[0] - self.line_length * np.cos(image_yaw))
                # y1 = int(image_midpoint[1] - self.line_length * np.sin(image_yaw))

                cv2.line(img, (x0, y0), (x1, y1), self.vis_colors[i], 2)
                cv2.circle(img, (int(image_midpoint[0]), int(image_midpoint[1])), 5, self.vis_colors[i], -1)

        if self.tracked_wire_id is not None:
            tracked_midpoint = self.position_kalman_filters[self.tracked_wire_id].curr_pos
            image_midpoint = ct.world_to_image_pose(tracked_midpoint, pose_in_world, self.camera_vector)
            cv2.ellipse(img, (int(image_midpoint[0]), int(image_midpoint[1])), (15, 15), 0, 0, 360, (0, 255, 0), 3)
            
        return img
    
    def characterize_a_line(self, pt1, pt2, depth_image, segmentation_mask, pose):
        """
        Computes the average depth of points along a line that are also within the segmentation mask.
        

        Returns:
            mean_depth: The average depth of the points along the line that are within the segmentation mask.
        """
        # Generate points along the line using Bresenham's line algorithm
        line_points = np.array(cv2.line(np.zeros_like(depth_image, dtype=np.uint8), pt1, pt2, 1))
        y_indices, x_indices = np.where(line_points == 1)
        
        # Filter points that fall within the segmentation mask
        valid_mask = segmentation_mask[y_indices, x_indices] > 0
        valid_xs = x_indices[valid_mask]
        valid_ys = y_indices[valid_mask]
        valid_depths = depth_image[y_indices[valid_mask], x_indices[valid_mask]]

        # remove nans or infs or 0s
        valid_depths = valid_depths[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]
        valid_xs = valid_xs[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]
        valid_ys = valid_ys[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]

        if valid_depths.size > 0:
            world_points = ct.image_to_world_pose(np.hstack((valid_xs.reshape(-1, 1), valid_ys.reshape(-1, 1))), valid_depths, pose, self.camera_vector)
            avg_global_yaw = wd.compute_yaw_from_3D_points(world_points)

            cam_points_x, cam_points_y, cam_points_z = ct.image_to_camera(np.hstack((valid_xs.reshape(-1, 1), valid_ys.reshape(-1, 1))), valid_depths, self.camera_vector)

            avg_depth = np.mean(cam_points_z)
            return avg_depth, avg_global_yaw
        else:
            return None  # No valid depth values found
        
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
            self.declare_parameter('use_pose_cam', rclpy.Parameter.Type.BOOL)
            self.declare_parameter('max_distance_threshold', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('min_valid_kf_count_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('iteration_start_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('yaw_covariance', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('pos_covariance', rclpy.Parameter.Type.DOUBLE)

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

            self.use_pose_cam = self.get_parameter('use_pose_cam').get_parameter_value().bool_value
            self.distance_threshold = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
            self.valid_threshold = self.get_parameter('min_valid_kf_count_threshold').get_parameter_value().integer_value
            self.target_start_threshold = self.get_parameter('iteration_start_threshold').get_parameter_value().integer_value
            self.yaw_covariance = self.get_parameter('yaw_covariance').get_parameter_value().double_value
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