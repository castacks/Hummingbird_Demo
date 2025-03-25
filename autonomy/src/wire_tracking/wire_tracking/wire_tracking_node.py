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
        self.wire_detector = wd.WireDetector(threshold=self.line_threshold, 
                                             expansion_size=self.expansion_size, 
                                             low_canny_threshold=self.low_canny_threshold, 
                                             high_canny_threshold=self.high_canny_threshold,
                                             pixel_binning_size=self.pixel_binning_size,
                                             bin_avg_threshold_multiplier=self.bin_avg_threshold_multiplier)
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
            pose.orientation.y = - pose_msg.pose.orientation.y
            pose.orientation.z = - pose_msg.pose.orientation.z

        # camera frame from ros is x forward, y left, z up
        # self.get_logger().info(f"Wire Cam Position: {pose.position.x}, {pose.position.y}, {pose.position.z}")

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
        debug_image = None
        if np.any(seg_mask):
            wire_lines, wire_midpoints, wire_yaw_in_image = self.wire_detector.detect_wires(seg_mask)

            # get the horizontal camera yaw
            if len(wire_midpoints) != 0:
                cos_yaw = 0.0
                sin_yaw = 0.0
                corresponding_depths = np.zeros(len(wire_midpoints))
                for i, (x, y) in enumerate(wire_midpoints):

                    # depth_midpoint = depth[y, x]
                    # find all depths that are in the segmentation mask and lie on the line of the wire, and average their depths
                    pt1 = np.array([wire_lines[i][0], wire_lines[i][1]])
                    pt2 = np.array([wire_lines[i][2], wire_lines[i][3]])
                    wire_depth, wire_global_yaw = self.characterize_a_line(pt1, pt2, depth, seg_mask, pose)
                    if wire_depth is None or wire_global_yaw is None:
                        continue
                    
                    cos_yaw += np.cos(wire_global_yaw)
                    sin_yaw += np.sin(wire_global_yaw)

                    corresponding_depths[i] = wire_depth

                if corresponding_depths.size != 0 or (cos_yaw != 0.0 and sin_yaw != 0.0):
                    global_midpoints = ct.image_to_world_pose(wire_midpoints, corresponding_depths, pose, self.camera_vector)
                    cos_yaw /= len(wire_midpoints)
                    sin_yaw /= len(wire_midpoints)
                    global_yaw = wd.clamp_angles_pi(np.arctan2(sin_yaw, cos_yaw))

                if self.are_kfs_initialized == False:
                    self.initialize_kfs(global_midpoints, global_yaw)
                else:
                    self.update_kfs(global_midpoints, global_yaw, pose)

                # self.debug_kfs(detected_midpoints=wire_midpoints, depth_midpoints=corresponding_depths)

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
                debug_image = self.draw_kfs_viz(image, pose, wire_lines)
            else:
                debug_image = None        
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
                
    def characterize_a_line(self, pt1, pt2, depth_image, segmentation_mask, pose):
        """
        Computes the average depth of points along a line that are also within the segmentation mask.
        
        Returns:
            mean_depth: The average depth of the points along the line that are within the segmentation mask.
        """
        # Generate points along the line using Bresenham's line algorithm
        line_mask = np.zeros_like(depth_image, dtype=np.uint8)
        if isinstance(pt1, np.ndarray):
            pt1 = (pt1[0], pt1[1])
        if isinstance(pt2, np.ndarray):
            pt2 = (pt2[0], pt2[1])
        cv2.line(line_mask, pt1, pt2, 1)
        y_indices, x_indices = np.where(line_mask == 1)
        
        # Filter points that fall within the segmentation mask
        valid_mask = segmentation_mask[y_indices, x_indices] > 0
        valid_xs = x_indices[valid_mask]
        valid_ys = y_indices[valid_mask]
        valid_depths = depth_image[y_indices[valid_mask], x_indices[valid_mask]]

        # remove nans or infs or 0s
        line_depths = valid_depths[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]
        line_xs = valid_xs[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]
        line_ys = valid_ys[~np.isnan(valid_depths) & ~np.isinf(valid_depths) & (valid_depths > 0.0)]

        if line_depths.size > 0:
            image_points = np.hstack((line_xs.reshape(-1, 1), line_ys.reshape(-1, 1)))

            world_points = ct.image_to_world_pose(image_points, line_depths, pose, self.camera_vector)
            assert world_points.shape[0] == line_depths.size
            avg_global_yaw = wd.compute_yaw_from_3D_points(world_points)

            cam_points_x, cam_points_y, cam_points_z = ct.image_to_camera(np.hstack((line_xs.reshape(-1, 1), line_ys.reshape(-1, 1))), line_depths, self.camera_vector)

            avg_depth = np.mean(cam_points_z)
            return avg_depth, avg_global_yaw
        else:
            return None, None  # No valid depth values found
        
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
            self.declare_parameter('low_canny_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('high_canny_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('pixel_binning_size', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('bin_avg_threshold_multiplier', rclpy.Parameter.Type.DOUBLE)

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
            self.low_canny_threshold = self.get_parameter('low_canny_threshold').get_parameter_value().integer_value
            self.high_canny_threshold = self.get_parameter('high_canny_threshold').get_parameter_value().integer_value
            self.pixel_binning_size = self.get_parameter('pixel_binning_size').get_parameter_value().integer_value
            self.bin_avg_threshold_multiplier = self.get_parameter('bin_avg_threshold_multiplier').get_parameter_value().double_value

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