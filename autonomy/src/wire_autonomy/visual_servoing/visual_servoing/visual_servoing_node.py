#!/usr/bin/env python

import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
import yaml
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from geometry_msgs.msg import TwistStamped, PoseStamped
from visualization_msgs.msg import Marker

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

# For synchronized message filtering
from message_filters import ApproximateTimeSynchronizer, Subscriber

from wire_interfaces.msg import WireTarget

from ament_index_python.packages import get_package_share_directory

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

        self.delta_theta = 0.1
        self.desired_theta = 0.0
        self.desired_rho = 0.0

        # Subscribers
        self.wire_target_sub = self.create_subscription(WireTarget, self.wire_target_pub_topic, self.input_callback, 1)

        # Visual Servoing timer
        visual_servo_callback_group = MutuallyExclusiveCallbackGroup()
        self.visual_servo_timer = self.create_timer(0.1, self.ibvs_control, callback_group=visual_servo_callback_group)

        # Service
        activate_srv_cb_group = ReentrantCallbackGroup()
        self.activate_srv = self.create_service(Trigger, self.activate_srv_topic, self.activate_callback, callback_group=activate_srv_cb_group)

        # Publishers
        self.velocity_pub = self.create_publisher(TwistStamped, self.velocity_pub_topic, 1)

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
        
    def publish_velocity_ibvs(self, vy, vz, v_yaw):
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = "/drone"
        vel_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        vel_msg.twist.linear.y = vy
        vel_msg.twist.linear.z = vz
        vel_msg.twist.angular.z = v_yaw
        self.velocity_pub.publish(vel_msg)

    def ibvs_control_plucker(self, theta0, rho0, pitch, center_point):
        if self.activate_wire_grasping:
            theta1 = theta0 + self.delta_theta
            theta2 = theta0 - self.delta_theta
            x1, y1 = center_point[0] + rho0 * np.cos(theta1), center_point[1] + rho0 * np.sin(theta1)
            x2, y2 = center_point[0] + rho0 * np.cos(theta2), center_point[1] + rho0 * np.sin(theta2)
            z1 = center_point[2] + pitch * np.linalg.norm(center_point[:2] - np.array([x1, y1]))
            z2 = center_point[2] + pitch * np.linalg.norm(center_point[:2] - np.array([x2, y2]))
            rho1 = rho0 / (np.cos(theta1) * np.cose(theta0) + np.sin(theta1) * np.sin(theta0))
            rho2 = rho0 / (np.cos(theta2) * np.cos(theta0) + np.sin(theta2) * np.sin(theta0))
            z0 = center_point[2]
            assert rho1 == rho2, "rho1 and rho2 should be equal, got {}, {}".format(rho1, rho2)
            L_theta_vx = (1 / (2 * rho1)) * (np.cos(theta1) / z1 - np.cos(theta2) / z2) * np.cot(self.delta_theta) +  (1 / (2 * rho1)) * (np.sin(theta2) / z2 + np.sin(theta1) / z1)
            L_theta_vy = (1 / (2 * rho1)) * (np.sin(theta1) / z1 - np.sin(theta2) / z2) * np.cot(self.delta_theta) +  (1 / (2 * rho1)) * (np.cos(theta2) / z2 + np.cos(theta1) / z1)
            L = np.array([
                [ - np.cos(theta0) / z0, - np.sin(theta0) / z0, rho0 / z0],
                [L_theta_vx, L_theta_vy, 1 / (2 * z2) - 1 / (2 * z1)]
            ])
            error_theta = theta0 - self.desired_theta
            error_rho = rho0 - self.desired_rho
            e = np.array([[error_theta], [error_rho]])
            L_inv = np.linalg.pinv(L)
            v = - np.dot(L_inv, e)
            vx, vy, _ = v.flatten()
            vz = center_point[2] - self.z_wire_offset_from_camera_m
            





    def ibvs_control_point(self, x, y, z, pose):
            image_yaw = self.get_image_angle_from_kfs(pose)
            if x > 0 and x < self.cx * 2 and y > 0 and y < self.cy * 2:
                curr_z = pose.position.z

                # Compute error vector in image plane
                e_u = self.cy - y
                e_v = self.cx - x
                e = np.array([[e_u], [e_v]])
                # Compute interaction matrix for translation only
                L = np.array([
                                [-self.fy/z, 0, y/z],
                                [0, -self.fx/z, x/z]
                            ])
                # Compute the pseudo-inverse of the interaction matrix
                L_inv = np.linalg.pinv(L)

                # lambda_gain = 1.0
                y_gain = 0.6
                z_gain = 1.75
                yaw_gain = -0.25

                z_stop = 0.85
                z_offset = 1.25
                delta_z = z - z_offset - curr_z
                vz = z_gain * delta_z
                vz = min(vz, self.v_z_limit) # set the limit for the z velocity

                # prevents stability issues when the wire is too close to the camera
                if delta_z < z_stop:
                    y_gain = 0.2

                v_c = - np.dot(L_inv, e)
                vy, vx, _ = v_c.flatten()
                vy = y_gain * vy

                delta_angle = (image_yaw + np.pi) % (2 * np.pi) - np.pi
                v_yaw = delta_angle * yaw_gain

            else:
                vx, vy, vz, v_yaw = 0.0, 0.0, 0.0, 0.0
        
            self.publish_velocity_ibvs(vy, vz, v_yaw)
        
    def set_params(self):
        try:
            # Services
            self.activate_srv_topic = self.get_parameter('activate_srv_topic').get_parameter_value().string_value

            # Subscribers
            self.declare_parameter('wire_target_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_target_pub_topic = self.get_parameter('wire_target_pub_topic').get_parameter_value().string_value            

            # Publishers
            self.declare_parameter('velocity_pub_topic', rclpy.Parameter.Type.STRING)
            self.velocity_pub_topic = self.get_parameter('velocity_pub_topic').get_parameter_value().string_value

            with open(get_package_share_directory('visual_servoing') + '/config/visual_servo_config.yaml', 'r') as file:
                self.visual_servo_config = yaml.safe_load(file)

            self.y_wire_offset_from_camera_m = self.visual_servo_config['y_wire_offset_from_camera_m']
            self.z_wire_offset_from_camera_m = self.visual_servo_config['z_wire_offset_from_camera_m']

            self.kp_y_value = self.visual_servo_config['kp_y_value']
            self.kd_y_value = self.visual_servo_config['kd_y_value']
            self.ki_y_value = self.visual_servo_config['ki_y_value']
            self.v_y_limit = self.visual_servo_config['v_y_limit']

            self.kp_z_value = self.visual_servo_config['kp_z_value']
            self.kd_z_value = self.visual_servo_config['kd_z_value']
            self.ki_z_value = self.visual_servo_config['ki_z_value']
            self.v_z_limit = self.visual_servo_config['v_z_limit']

            self.kp_yaw_value = self.visual_servo_config['kp_yaw_value']
            self.kd_yaw_value = self.visual_servo_config['kd_yaw_value']
            self.ki_yaw_value = self.visual_servo_config['ki_yaw_value']
            self.v_yaw_limit = self.visual_servo_config['v_yaw_limit']

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
    