#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import time

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from wire_interfaces.msg import WireTarget
from mavros_msgs.msg import PositionTarget

from ament_index_python.packages import get_package_share_directory

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def clamp_angle_rad(angle):
    """
    Clamp an angle in radians to the range [-pi, pi].
    Works with scalars or numpy arrays.
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class WireGraspingNode(Node):
    def __init__(self):
        super().__init__('wire_grasping_node')
        self.set_params()

        # Subscribers
        self.wire_target_sub = self.create_subscription(WireTarget, self.wire_target_topic, self.wire_target_callback, 1)

        # Publishers
        self.velocity_pub = self.create_publisher(PositionTarget, self.velocity_cmd_topic, 1)

        self.first_pass = True
        self.get_logger().info("Wire Grasping Node initialized")
        
        
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
        
    def publish_velocity(self, vx, vy, vz, v_yaw):
        vel_target_msg = PositionTarget()
        vel_target_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        vel_target_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW
        vel_target_msg.velocity.x = np.clip(vx, -self.v_xy_limit, self.v_xy_limit)
        vel_target_msg.velocity.y = np.clip(vy, -self.v_xy_limit, self.v_xy_limit)
        vel_target_msg.velocity.z = np.clip(vz, -self.v_z_limit, self.v_z_limit)
        vel_target_msg.yaw_rate = np.clip(v_yaw, -self.v_yaw_limit, self.v_yaw_limit)
        self.get_logger().info(f"Sending velocity command: {vx} m/s, {vy} m/s, {vz} m/s, {v_yaw} rad/s")
        self.velocity_pub.publish(vel_target_msg)

    def wire_target_callback(self, msg):
        self.prev_err_x = getattr(self, "prev_err_x", 0)
        self.prev_err_y = getattr(self, "prev_err_y", 0)
        self.prev_err_z = getattr(self, "prev_err_z", 0)
        self.prev_err_yaw = getattr(self, "prev_err_yaw", 0)
        self.sum_err_x = getattr(self, "sum_err_x", 0)
        self.sum_err_y = getattr(self, "sum_err_y", 0)
        self.sum_err_z = getattr(self, "sum_err_z", 0)
        self.sum_err_yaw = getattr(self, "sum_err_yaw", 0)

        target_position = np.array([msg.target_x, msg.target_y - self.x_wire_offset_from_camera_m, msg.target_z - self.z_wire_offset_from_camera_m])
        target_yaw = msg.target_yaw

        if self.first_pass is True:
            self.vx = self.kp_xy * target_position[0]
            self.vy = self.kp_xy * target_position[1]
            self.vz = self.kp_z * target_position[2]
            self.yaw_rate = self.kp_yaw * target_yaw
            dt = 0.0
            self.first_pass = False
        else:
            dt = time.perf_counter() - self.prev_timestamp
            self.vx = self.kp_xy * target_position[0] + self.kd_xy * (target_position[0] - self.prev_err_x) / dt + self.ki_xy * self.sum_err_x
            self.vy = self.kp_xy * target_position[1] + self.kd_xy * (target_position[1] - self.prev_err_y) / dt + self.ki_xy * self.sum_err_y
            self.vz = self.kp_z * target_position[2] + self.kd_z * (target_position[2] - self.prev_err_z) / dt + self.ki_z * self.sum_err_z
            self.yaw_rate = self.kp_yaw * target_yaw + self.kd_yaw * clamp_angle_rad(target_yaw - self.prev_err_yaw) / dt + self.ki_yaw * self.sum_err_yaw

        self.prev_timestamp = time.perf_counter()
        self.prev_err_x = target_position[0]
        self.prev_err_y = target_position[1]
        self.prev_err_z = target_position[2]
        self.prev_err_yaw = target_yaw
        self.sum_err_x += target_position[0] * dt
        self.sum_err_y += target_position[1] * dt
        self.sum_err_z += target_position[2] * dt
        self.sum_err_yaw += target_yaw * dt

        self.publish_velocity(self.vx, self.vy, self.vz, self.yaw_rate)

    def set_params(self):
        try:
            # Services
            self.activate_srv_topic = self.get_parameter('activate_srv_topic').get_parameter_value().string_value

            # Subscribers
            self.declare_parameter('wire_target_topic', rclpy.Parameter.Type.STRING)
            self.wire_target_topic = self.get_parameter('wire_target_topic').get_parameter_value().string_value

            # Publishers
            self.declare_parameter('velocity_cmd_topic', rclpy.Parameter.Type.STRING)
            self.velocity_cmd_topic = self.get_parameter('velocity_cmd_topic').get_parameter_value().string_value

            with open(get_package_share_directory('visual_servoing') + '/config/visual_servo_config.yaml', 'r') as file:
                self.visual_servo_config = yaml.safe_load(file)

            self.x_wire_offset_from_camera_m = self.visual_servo_config['x_wire_offset_from_camera_m']
            self.z_wire_offset_from_camera_m = self.visual_servo_config['z_wire_offset_from_camera_m']

            self.kp_xy = self.visual_servo_config['kp_xy']
            self.kd_xy = self.visual_servo_config['kd_xy']
            self.ki_xy = self.visual_servo_config['ki_xy']
            self.v_xy_limit = self.visual_servo_config['v_xy_limit']

            self.kp_z = self.visual_servo_config['kp_z']
            self.kd_z = self.visual_servo_config['kd_z']
            self.ki_z = self.visual_servo_config['ki_z']
            self.v_z_limit = self.visual_servo_config['v_z_limit']

            self.kp_yaw = self.visual_servo_config['kp_yaw']
            self.kd_yaw = self.visual_servo_config['kd_yaw']
            self.ki_yaw = self.visual_servo_config['ki_yaw']
            self.v_yaw_limit = self.visual_servo_config['v_yaw_limit']

        except Exception as e:
            self.get_logger().error(f"Error in declare_parameters: {e}")
    
    
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
    