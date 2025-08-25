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
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

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
        self.wire_target_callback_group = MutuallyExclusiveCallbackGroup()
        self.wire_target_sub = self.create_subscription(WireTarget, self.wire_target_topic, self.wire_target_callback, 1)

        # Publishers
        self.velocity_pub = self.create_publisher(PositionTarget, self.velocity_cmd_topic, 1)

        self.velocity_timer = self.create_timer(1.0 / self.velocity_command_frequency, self.velocity_timer_callback)

        self.first_pass = True
        self.last_received_timestamp = None
        self.got_target = False

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.takeoff()
        self.get_logger().info("Servoing Node initialized and in flight")

    def call_service(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f'Waiting for {client.srv_name} service...')
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def takeoff(self):
        self.get_logger().info("Setting mode to GUIDED...")
        mode_request = SetMode.Request()
        mode_request.custom_mode = "GUIDED"
        mode_response = self.call_service(self.set_mode_client, mode_request)
        if not mode_response or not mode_response.mode_sent:
            self.get_logger().error("Failed to set GUIDED mode")
            return

        self.get_logger().info("Arming drone...")
        arm_request = CommandBool.Request()
        arm_request.value = True
        arm_response = self.call_service(self.arming_client, arm_request)
        if not arm_response or not arm_response.success:
            self.get_logger().error("Failed to arm the drone")
            return

        self.get_logger().info("Taking off...")
        takeoff_request = CommandTOL.Request()
        takeoff_request.altitude = self.takeoff_height
        takeoff_response = self.call_service(self.takeoff_client, takeoff_request)
        takeoff_skipped = False
        if not takeoff_response or not takeoff_response.success:
            self.get_logger().error("Failed to take off")
            takeoff_skipped = True

        if not takeoff_skipped:
            self.get_logger().info(f"Waiting {self.takeoff_wait_time} seconds before being ready to control...")
            time.sleep(self.takeoff_wait_time)
        return
        
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
        self.get_logger().info(f"Received wire target: {msg}")
        # Process the wire target message
        self.target_x = - msg.target_x - self.x_wire_offset_from_camera_m # flip for frame convention                                       
        self.target_y = - msg.target_y # flip for frame convention  
        self.target_z = msg.target_z - self.z_wire_offset_from_camera_m
        self.target_yaw = msg.target_yaw
        self.got_target = True
        self.last_received_timestamp = time.perf_counter()

    def velocity_timer_callback(self):
        if self.last_received_timestamp is not None and (time.perf_counter() - self.last_received_timestamp) < self.target_timeout and self.got_target:
            self.prev_err_x = getattr(self, "prev_err_x", 0)
            self.prev_err_y = getattr(self, "prev_err_y", 0)
            self.prev_err_z = getattr(self, "prev_err_z", 0)
            self.prev_err_yaw = getattr(self, "prev_err_yaw", 0)
            self.sum_err_x = getattr(self, "sum_err_x", 0)
            self.sum_err_y = getattr(self, "sum_err_y", 0)
            self.sum_err_z = getattr(self, "sum_err_z", 0)
            self.sum_err_yaw = getattr(self, "sum_err_yaw", 0)

            if self.first_pass is True:
                self.vx = self.kp_xy * self.target_x
                self.vy = self.kp_xy * self.target_y
                self.vz = self.kp_z * self.target_z
                self.yaw_rate = self.kp_yaw * self.target_yaw
                dt = 0.0
                self.first_pass = False
            else:
                dt = time.perf_counter() - self.prev_timestamp
                self.vx = self.kp_xy * self.target_x + self.kd_xy * (self.target_x - self.prev_err_x) / dt + self.ki_xy * self.sum_err_x
                self.vy = self.kp_xy * self.target_y + self.kd_xy * (self.target_y - self.prev_err_y) / dt + self.ki_xy * self.sum_err_y
                self.vz = self.kp_z * self.target_z + self.kd_z * (self.target_z - self.prev_err_z) / dt + self.ki_z * self.sum_err_z
                self.yaw_rate = self.kp_yaw * self.target_yaw + self.kd_yaw * clamp_angle_rad(self.target_yaw - self.prev_err_yaw) / dt + self.ki_yaw * self.sum_err_yaw

            self.prev_timestamp = time.perf_counter()
            self.prev_err_x = self.target_x
            self.prev_err_y = self.target_y
            self.prev_err_z = self.target_z
            self.prev_err_yaw = self.target_yaw
            self.sum_err_x += self.target_x * dt
            self.sum_err_y += self.target_y * dt
            self.sum_err_z += self.target_z * dt
            self.sum_err_yaw += self.target_yaw * dt

            self.publish_velocity(self.vx, self.vy, self.vz, self.yaw_rate)

        else:
            self.got_target = False
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)

    def set_params(self):
        try:
            self.declare_parameter('takeoff_height', 2.0)  # Height to take off to
            self.takeoff_height = self.get_parameter('takeoff_height').get_parameter_value().double_value
            self.declare_parameter('takeoff_wait_time', 5.0)  # Time to wait after takeoff before starting control
            self.takeoff_wait_time = self.get_parameter('takeoff_wait_time').get_parameter_value().double_value
            self.declare_parameter('target_timeout', 0.25)  # Time to wait before considering target lost
            self.target_timeout = self.get_parameter('target_timeout').get_parameter_value().double_value
            self.declare_parameter('velocity_command_frequency', 15.0)  # Hz
            self.velocity_command_frequency = self.get_parameter('velocity_command_frequency').get_parameter_value().double_value

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
    