#!/usr/bin/env python

from rcl_interfaces.msg import ParameterDescriptor
import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Transform, Pose
from cv_bridge import CvBridge

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ament_index_python.packages import get_package_share_directory

from .wire_detection import WireDetector, find_closest_point_on_3d_line, clamp_angles_pi, get_yaw_from_quaternion, get_distance_between_3D_point
from .kalman_filters import PositionKalmanFilter, YawKalmanFilter

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def transform_to_homogeneous(transform):
    """
    Converts a Transform message into a 4x4 homogeneous transformation matrix.

    Args:
        Transform (Transform): The ROS Transform message to convert.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    assert isinstance(transform, Transform) , "Input must be a Transform message, got %s" % type(Transform)
    # Extract translation components
    translation = transform.translation
    translation_vector = np.array([translation.x, translation.y, translation.z])

    # Extract rotation components
    rotation = transform.rotation
    quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Combine into a homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rot_matrix[:3, :3]
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix


class VisualServo(Node):
    def __init__(self):
        super().__init__('visual_servo_node')
        self.set_params()

        # Wire Detector
        self.wire_detector = WireDetector()
        self.position_kalman_filters = {}
        self.vis_colors = {}
        self.yaw_kalman_filter = None
        self.are_kfs_initialized = False
        self.max_kf_label = 0
        self.line_length = None
        self.tracked_wire_id = None
        self.total_iterations = 0

        # Subscribers
        self.received_camera_info = False
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1)
        self.rgb_image_sub = Subscriber(self, Image, self.rgb_image_sub_topic)
        self.depth_image_sub = Subscriber(self, Image, self.depth_image_sub_topic)
        self.pose_sub = Subscriber(self, PoseStamped, self.pose_sub_topic)
        self.bridge = CvBridge()

        # Time Synchronizer
        self.img_tss = ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub, self.pose_sub],
            queue_size=1, 
            slop=0.1
        )
        self.img_tss.registerCallback(self.image_callback)

        # Publishers
        self.tracking_viz_pub = self.create_publisher(Image, self.tracking_viz_pub_topic, 1)
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 1)

        self.get_logger().info("Visual Servo Node initialized")
        
    def camera_info_callback(self, data):
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.received_camera_info = True

    def image_callback(self, rgb_msg, depth_msg, pose_msg):
        if self.line_length is None:
            self.line_length = max(rgb_msg.width, rgb_msg.height) * 2
        try:
            # Convert the ROS image messages to OpenCV images
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        
        debug_image = None
        if self.received_camera_info:
            debug_image = self.detect_lines_and_update(rgb, depth, pose_msg.pose)
            if debug_image is not None:
                viz_pub_msg = self.bridge.cv2_to_imgmsg(debug_image, "rgb8")
                self.tracking_viz_pub.publish(viz_pub_msg) 
            # create depth visualization
            depth_viz = self.create_depth_viz(depth)
            depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, "rgb8")
            self.depth_viz_pub.publish(depth_viz_msg)

    def detect_lines_and_update(self, image, depth, pose: Pose):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, 50, 150, apertureSize=3)
        dilation_size = 10
        dilation_kernel = np.ones((dilation_size,dilation_size), np.uint8)
        seg_mask = cv2.dilate(seg_mask, dilation_kernel, iterations=1)
        seg_mask = cv2.erode(seg_mask, dilation_kernel, iterations=1)

        if np.any(seg_mask) and self.received_first_tf:
            wire_lines, wire_midpoints, avg_yaw = self.wire_detector.detect_wires(seg_mask)
            cam_yaw = get_yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            if len(wire_midpoints) != 0:
                global_yaw = cam_yaw + avg_yaw
                global_yaw = clamp_angles_pi(global_yaw) 
                corresponding_depths = np.array([])
                for i, (x, y) in enumerate(wire_midpoints):
                    corresponding_depths = np.append(corresponding_depths, depth[y, x])
                
                transform = Transform()
                transform.translation.x = pose.position.x
                transform.translation.y = pose.position.y
                transform.translation.z = pose.position.z
                transform.rotation.x = pose.orientation.x
                transform.rotation.y = pose.orientation.y
                transform.rotation.z = pose.orientation.z
                transform.rotation.w = pose.orientation.w
                global_midpoints = self.image_to_world(wire_midpoints, corresponding_depths, transform)
                if self.are_kfs_initialized == False:
                    self.initialize_kfs(global_midpoints, global_yaw)
                else:
                    self.update_kfs(global_midpoints, global_yaw)

            # debug_img = self.draw_wire_lines(debug_img, wire_lines, wire_midpoints)
            # self.debug_kfs()
            # currently choosing the tracked wire is based on valid count, should be based on distance from the center of the image
            self.total_iterations += 1
            if self.tracked_wire_id is None and self.total_iterations > self.target_start_threshold:
                min_distance = 100000
                min_id = None
                for i, kf in self.position_kalman_filters.items():
                    if kf.valid_count >= self.valid_threshold:
                        curr_pos = transform.translation
                        distance = get_distance_between_3D_point(kf.curr_pos, np.array([curr_pos.x, curr_pos.y, curr_pos.z]))
                        if distance < min_distance:
                            min_distance = distance
                            min_id = i
                self.tracked_wire_id = min_id
            if self.are_kfs_initialized:
                debug_img = self.draw_valid_kfs(image, transform)
            return debug_img

    def initialize_kfs(self, global_midpoints, global_yaw):
        if not self.are_kfs_initialized:
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
                        # self.get_logger().info(f'distance: {distance}, mid: {mid}, merged_mid: {merged_mid}')
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
        
        
        self.vis_colors[self.max_kf_label] = np.random.randint(0, 256, 3).tolist()

    def debug_kfs(self, detected_midpoints = None):
        valid_counts = []
        midpoints = np.array([kf.curr_pos for kf in self.position_kalman_filters.values()])
        for i, kf in self.position_kalman_filters.items():
            valid_counts.append(kf.valid_count)
        if detected_midpoints is not None:
            self.get_logger().info("Detected Midpoints: %s" % str(detected_midpoints))
        self.get_logger().info("Num KFs: %s" % len(self.position_kalman_filters))
        self.get_logger().info("Valid Counts: %s" % str(valid_counts))

        # self.get_logger().info("Midpoints: %s" % str(midpoints))

    def update_kfs(self, new_global_midpoints, global_yaw):
        matched_indices = []
        new_midpoint = []
        self.yaw_kalman_filter.update(global_yaw)
        for i, mid in enumerate(new_global_midpoints):
            got_matched = False
            for j, kf in self.position_kalman_filters.items():
                closest_point = find_closest_point_on_3d_line(mid, self.yaw_kalman_filter.curr_yaw, kf.curr_pos)
                distance = np.linalg.norm(closest_point - kf.curr_pos, ord=2)

                # if the closest point on the detected line is close enough to an existing Kalman Filter, update it
                if distance < self.distance_threshold:
                    kf.update(mid)
                    matched_indices.append(i)
                    got_matched = True
                    break
            if not got_matched:
                new_midpoint.append(mid)

        # Add new Kalman Filters for unmatched midpoints
        # self.get_logger().info(f'adding {len(new_midpoint)} new Kalman Filters')
        for mid in new_midpoint:
            self.add_kf(mid)

        # Remove Kalman Filters that have not been updated for a while
        kfs_to_remove = []
        for i, kf in self.position_kalman_filters.items():
            if i not in matched_indices:
                x_img, y_img = self.world_to_image(kf.curr_pos[0], kf.curr_pos[1], kf.curr_pos[2], self.tf_camera_to_world)
                if x_img > 0 and x_img < self.cx * 2 and y_img > 0 and y_img < self.cy * 2:
                    kf.valid_count -= 1
                if kf.valid_count < 0:
                    kfs_to_remove.append(i)
        for i in kfs_to_remove:
            if i == self.tracked_wire_id:
                self.tracked_wire_id = None
                self.get_logger().info("Had to remove tracked wire")
            self.position_kalman_filters.pop(i)
            self.vis_colors.pop(i)

    def predict_kfs(self):
        for i, kf in self.position_kalman_filters.items():
            kf.predict()
        self.yaw_kalman_filter.predict()

    def image_to_world(self, points, depth, tf_camera_to_world):
        x_c, y_c, z_c = self.image_to_camera(points, depth.reshape(-1, 1))

        H_cam_to_world = transform_to_homogeneous(tf_camera_to_world)

        # Convert the point to a numpy array
        point_vec = np.hstack((x_c, y_c, z_c, np.ones_like(x_c)))

        # Apply the transform: Rotate then translate
        with np.errstate(invalid='ignore'):
            try:
                world_points = H_cam_to_world @ point_vec.T
            except Exception as e:
                self.get_logger().info(f"Error: {H_cam_to_world}, {point_vec.T}")

        return world_points[:3].T

    def image_to_camera(self, image_points, depth):
        camera_x = (image_points[:,0] - self.cx).reshape(-1, 1) * depth / self.fx
        camera_y = (image_points[:,1] - self.cy).reshape(-1, 1) * depth / self.fy
        return camera_x, camera_y, depth

    def camera_to_image(self, camera_x, camera_y, camera_z):
        image_x = (camera_x * self.fx / camera_z) + self.cx
        image_y = (camera_y * self.fy / camera_z) + self.cy
        return image_x, image_y

    def world_to_image(self, world_x, world_y, world_z, tf_camera_to_world):

        H_cam_to_world = transform_to_homogeneous(tf_camera_to_world)
        H_cam_to_world_inv = np.linalg.inv(H_cam_to_world)

        # Convert the point to a numpy array
        point_vec = np.array([world_x, world_y, world_z])

        # Apply the transform: Rotate then translate
        cam_point = H_cam_to_world_inv @ np.append(point_vec, 1)

        return self.camera_to_image(cam_point[0], cam_point[1], cam_point[2])

    def draw_valid_kfs(self, img, tf_cam_to_world):
        global_yaw = self.yaw_kalman_filter.curr_yaw
        cam_yaw = get_yaw_from_quaternion(tf_cam_to_world.rotation.x, tf_cam_to_world.rotation.y, tf_cam_to_world.rotation.z, tf_cam_to_world.rotation.w)
        image_yaw = global_yaw - cam_yaw
        for i, kf in self.position_kalman_filters.items():
            if kf.valid_count >= self.valid_threshold:
                image_midpoint = self.world_to_image(kf.curr_pos[0], kf.curr_pos[1], kf.curr_pos[2], tf_cam_to_world)
                x0 = int(image_midpoint[0] + self.line_length * np.cos(image_yaw))
                y0 = int(image_midpoint[1] + self.line_length * np.sin(image_yaw))
                x1 = int(image_midpoint[0] - self.line_length * np.cos(image_yaw))
                y1 = int(image_midpoint[1] - self.line_length * np.sin(image_yaw))
                cv2.line(img, (x0, y0), (x1, y1), self.vis_colors[i], 2)
                cv2.circle(img, (int(image_midpoint[0]), int(image_midpoint[1])), 5, self.vis_colors[i], -1)

        if self.tracked_wire_id is not None:
            tracked_midpoint = self.position_kalman_filters[self.tracked_wire_id].curr_pos
            image_midpoint = self.world_to_image(tracked_midpoint[0], tracked_midpoint[1], tracked_midpoint[2], tf_cam_to_world)
            cv2.ellipse(img, (int(image_midpoint[0]), int(image_midpoint[1])), (15, 15), 0, 0, 360, (0, 255, 0), 3)
        return img
    
    def draw_wire_lines(self, img, wire_lines, wire_midpoints, center_line=None, center_line_midpoint=None):
        for i, (x, y) in enumerate(wire_midpoints):
            x0, y0, x1, y1 = wire_lines[i]
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        if center_line is not None:
            x0, y0, x1, y1 = center_line
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(img, (int(center_line_midpoint[0]), int(center_line_midpoint[1])), 5, (0, 0, 255), -1)
        return img
    
    def create_depth_viz(self, depth):
        depth_viz = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        return depth_viz
        
    def set_params(self):
        try:
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('pose_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('tracking_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('max_distance_threshold', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('min_valid_kf_count_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('iteration_start_threshold', rclpy.Parameter.Type.INTEGER)

            # Access parameters
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value
            self.pose_sub_topic = self.get_parameter('pose_sub_topic').get_parameter_value().string_value
            self.tracking_viz_pub_topic = self.get_parameter('tracking_viz_pub_topic').get_parameter_value().string_value
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.use_segmentation = self.get_parameter('use_segmentation').get_parameter_value().bool_value
            self.distance_threshold = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
            self.valid_threshold = self.get_parameter('min_valid_kf_count_threshold').get_parameter_value().integer_value
            self.target_start_threshold = self.get_parameter('iteration_start_threshold').get_parameter_value().integer_value
        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    

def main():
    rclpy.init()
    node = VisualServo()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()