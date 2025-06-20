#!/usr/bin/env python

import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped, PoseStamped
from visualization_msgs.msg import Marker

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireGraspingNode(Node):
    def __init__(self):
        super().__init__('wire_grasping_node')
        self.set_params()

        # Toggles
        self.activate_wire_grasping = False
        self.initialized = False

        # Subscribers
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1)

        # Visual Servoing timer
        visual_servo_cb_group = ReentrantCallbackGroup()
        self.visual_servo_timer = self.create_timer(0.1, self.ibvs_control, callback_group=visual_servo_cb_group)

        # Service
        activate_srv_cb_group = ReentrantCallbackGroup()
        self.activate_srv = self.create_service(Trigger, self.activate_srv_topic, self.activate_callback, callback_group=activate_srv_cb_group)

        # Publishers
        self.velocity_pub = self.create_publisher(TwistStamped, self.velocity_pub_topic, 1)
        self.tracking_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 10)

        self.get_logger().info("Wire Grasping Node initialized")
        
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
        self.get_logger().info("Visual Servoing Node initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)

    def input_callback(self, rgb_msg, depth_msg):
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
        
        try:
            transform = self.tf_buffer.lookup_transform(self.world_frame_id, self.camera_frame_id, rgb_msg.header.stamp).transform
            self.received_first_tf = True
        except Exception as e:
            self.get_logger().info(f"TF lookup failed: {e}")
            return
        pose = ct.tf_to_pose(transform)

        debug_image = None
        if self.received_camera_info and self.activate_wire_grasping and self.received_first_tf:
            # transform pose cam pose to wire cam pose
            debug_image = self.detect_lines_and_update(rgb, depth, pose)
            self.ibvs_control(pose)

        if debug_image is None:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(rgb, "rgb8")
        else:
            viz_pub_msg = self.bridge.cv2_to_imgmsg(debug_image, "rgb8")
        self.tracking_viz_pub.publish(viz_pub_msg) 
        
    def activate_callback(self, request, response):
        if request.data:
            self.activate_wire_grasping = True
            response.success = True
            response.message = "Wire grasping activated."
            self.get_logger().info("Wire grasping activated.")
        else:
            self.activate_wire_grasping = False
            response.success = True
            response.message = "Wire grasping deactivated."
            self.get_logger().info("Wire grasping deactivated.")
        return response
        
    def publish_velocity_ibvs(self, vx, vy, vz, v_yaw):
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/drone"
        vel_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        vel_msg.twist.linear.x = vx
        vel_msg.twist.linear.y = vy
        vel_msg.twist.linear.z = vz
        vel_msg.twist.angular.z = v_yaw
        vel_msg.twist.angular.x = 0.0
        vel_msg.twist.angular.y = 0.0
        self.velocity_pub.publish(vel_msg)

    def ibvs_control(self, pose):
            x_w, y_w, z_w = self.position_kalman_filters[self.tracked_wire_id].curr_pos
            x_c, y_c, z_c = ct.world_to_camera(np.array([[x_w, y_w, z_w]]), pose, self.camera_vector)
            image_yaw = self.get_image_angle_from_kfs(pose)

            x, y = ct.world_to_image(np.array([[x_w, y_w, z_w]]), pose, self.camera_vector)
            if x > 0 and x < self.cx * 2 and y > 0 and y < self.cy * 2:
                curr_z = pose.position.z

                # Compute error vector in image plane
                e_u = self.cy - y
                e_v = self.cx - x
                e = np.array([[e_u], [e_v]])
                # Compute interaction matrix for translation only
                L = np.array([
                                [-self.fy/z_c, 0, y/z_c],
                                [0, -self.fx/z_c, x/z_c]
                            ])
                # Compute the pseudo-inverse of the interaction matrix
                L_inv = np.linalg.pinv(L)

                # lambda_gain = 1.0
                x_gain = 0.6
                y_gain = 0.6
                z_gain = 1.75
                yaw_gain = -0.25

                z_stop = 0.85
                z_offset = 1.25
                z_limit = 12.0
                delta_z = z_c - z_offset - curr_z
                vz = z_gain * delta_z
                vz = min(vz, z_limit) # set the limit for the z velocity

                # prevents stability issues when the wire is too close to the camera
                if delta_z < z_stop:
                    x_gain = 0.2
                    y_gain = 0.2

                v_c = - np.dot(L_inv, e)
                vy, vx, _ = v_c.flatten()
                vx = x_gain * vx
                vy = y_gain * vy

                old_vx = vx
                old_vy = vy
                vx = old_vy
                vy = -old_vx

                delta_angle = (image_yaw + np.pi) % (2 * np.pi) - np.pi
                v_yaw = delta_angle * yaw_gain

            else:
                vx, vy, vz, v_yaw = 0.0, 0.0, 0.0, 0.0
        
            self.publish_velocity_ibvs(vx, vy, vz, v_yaw)
        
    def set_params(self):
        try:
            # Access parameters
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value

            self.activate_srv_topic = self.get_parameter('activate_srv_topic').get_parameter_value().string_value

            self.wire_viz_pub_topic = self.get_parameter('wire_viz_pub_topic').get_parameter_value().string_value
            self.velocity_pub_topic = self.get_parameter('velocity_pub_topic').get_parameter_value().string_value

            self.line_threshold = self.get_parameter('line_threshold').get_parameter_value().integer_value
            self.expansion_size = self.get_parameter('expansion_size').get_parameter_value().integer_value

            self.world_frame_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
            self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value
            self.tf_update_rate = self.get_parameter('tf_update_rate').get_parameter_value().double_value

            self.distance_threshold = self.get_parameter('max_distance_threshold').get_parameter_value().double_value
            self.valid_threshold = self.get_parameter('min_valid_kf_count_threshold').get_parameter_value().integer_value
            self.target_start_threshold = self.get_parameter('iteration_start_threshold').get_parameter_value().integer_value
            self.yaw_covariance = self.get_parameter('yaw_covariance').get_parameter_value().double_value
            self.pos_covariance = self.get_parameter('pos_covariance').get_parameter_value().double_value

        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    
    
def main():
    rclpy.init()
    node = WireGraspingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    