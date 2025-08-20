#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, LivelinessPolicy
from rclpy.duration import Duration
import time
from geometry_msgs.msg import TwistStamped, PoseStamped
import numpy as np
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

class WireTransforms:
    def __init__(self, wire_x_offset, wire_y_offset, wire_z_offset, wire_yaw_offset, drone_origin_position, drone_origin_quat):
        self.drone_origin_yaw = R.from_quat(drone_origin_quat).as_euler('zyx')[0]
        self.rot_matrix = np.array([[np.cos(self.drone_origin_yaw), -np.sin(self.drone_origin_yaw)],
                                     [np.sin(self.drone_origin_yaw), np.cos(self.drone_origin_yaw)]])
        x_offset, y_offset = self.rot_matrix @ np.array([wire_x_offset, wire_y_offset])
        self.wire_x = drone_origin_position[0] + x_offset
        self.wire_y = drone_origin_position[1] + y_offset
        self.wire_z = drone_origin_position[2] + wire_z_offset
        self.wire_yaw = self.drone_origin_yaw + wire_yaw_offset
        self.wire_pos = np.array([self.wire_x, self.wire_y, self.wire_z])

    def transform_wire_local_to_body(self, drone_pos, drone_quat):
        """
        Transform wire pose from local ENU frame to drone body frame (base_link/FLU).

        Args:
            drone_pos (array-like): [x, y, z] drone position in local ENU
            drone_quat (array-like): [x, y, z, w] quaternion of drone in local ENU
            wire_pos (array-like): [x, y, z] wire point in local ENU
            wire_yaw (float): yaw angle of the wire in local ENU

        Returns:
            wire_pos_body (np.ndarray): wire point in body frame (3,)
            wire_dir_body (np.ndarray or None): wire direction in body frame (3,) if given
        """
        drone_pos = np.array(drone_pos)

        # Rotation from world->body
        R_wb = R.from_quat(drone_quat).as_matrix()
        R_bw = R_wb.T  # inverse rotation

        # Translate wire into drone-centered coords
        delta_w = self.wire_pos - drone_pos
        wire_pos_body = R_bw @ delta_w

        wire_dir_w = np.array([np.cos(self.wire_yaw), np.sin(self.wire_yaw), 0.0])

        # Rotate direction into body frame
        wire_dir_b = R_bw @ wire_dir_w

        # Compute yaw of wire in body frame
        wire_yaw_body = np.arctan2(wire_dir_b[1], wire_dir_b[0])
        wire_yaw_body = self.clamp_angle_rad(wire_yaw_body)

        return wire_pos_body, wire_yaw_body
    
    def clamp_angle_rad(self, angle):
        """
        Clamp an angle in radians to the range [-pi, pi].
        Works with scalars or numpy arrays.
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

class DroneControlNode(Node):
    def __init__(self):
        super().__init__('wire_control_node')

        # Get parameters
        self.declare_parameter('max_velocity_xy', 0.5)  # m/s
        self.declare_parameter('max_velocity_z', 0.5)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('takeoff_height', 2.0)  # Height to take off to
        self.declare_parameter('takeoff_wait_time', 5.0)  # Time to wait after takeoff before starting control

        self.declare_parameter('wire_yaw_offset', 0.0)  # Yaw angle of the wire
        self.declare_parameter('wire_x_offset', 0.0)  # X coordinate of the wire
        self.declare_parameter('wire_y_offset', 0.0)  # Y coordinate of the wire
        self.declare_parameter('wire_z_offset', 6.0)  # Z coordinate of the wire

        self.declare_parameter('p_xy', 0.5)
        self.declare_parameter('d_xy', 0.1)
        self.declare_parameter('i_xy', 0.0)
        self.declare_parameter('p_z', 0.25)
        self.declare_parameter('d_z', 0.05)
        self.declare_parameter('i_z', 0.0)
        self.declare_parameter('p_yaw', 1.0)
        self.declare_parameter('d_yaw', 0.1)
        self.declare_parameter('i_yaw', 0.0)

        self.control_rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self.max_velocity_xy = self.get_parameter('max_velocity_xy').get_parameter_value().double_value
        self.max_velocity_z = self.get_parameter('max_velocity_z').get_parameter_value().double_value
        self.takeoff_wait_time = self.get_parameter('takeoff_wait_time').get_parameter_value().double_value
        self.takeoff_height = self.get_parameter('takeoff_height').get_parameter_value().double_value

        self.wire_yaw_offset = self.get_parameter('wire_yaw_offset').get_parameter_value().double_value
        self.wire_x_offset = self.get_parameter('wire_x_offset').get_parameter_value().double_value
        self.wire_y_offset = self.get_parameter('wire_y_offset').get_parameter_value().double_value
        self.wire_z_offset = self.get_parameter('wire_z_offset').get_parameter_value().double_value

        self.p_xy = self.get_parameter('p_xy').get_parameter_value().double_value
        self.d_xy = self.get_parameter('d_xy').get_parameter_value().double_value
        self.i_xy = self.get_parameter('i_xy').get_parameter_value().double_value
        self.p_z = self.get_parameter('p_z').get_parameter_value().double_value
        self.d_z = self.get_parameter('d_z').get_parameter_value().double_value
        self.i_z = self.get_parameter('i_z').get_parameter_value().double_value
        self.p_yaw = self.get_parameter('p_yaw').get_parameter_value().double_value
        self.d_yaw = self.get_parameter('d_yaw').get_parameter_value().double_value
        self.i_yaw = self.get_parameter('i_yaw').get_parameter_value().double_value

        # MAVROS service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.pos_control_publisher = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,   # UNKNOWN not valid → fallback to KEEP_LAST
            depth=1,                           # Required with KEEP_LAST, choose any small positive integer
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )

        self.position_subscriber = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.position_estimate_callback,
            qos_profile
        )

        # Start takeoff to get drone into starting position
        self.origin_position = None
        self.prev_timestamp = None
        self.first_pass = True
        self.takeoff()

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

    def position_estimate_callback(self, msg):
        if self.origin_position is None:
            self.origin_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            self.origin_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
            self.wire_transforms = WireTransforms(self.wire_x_offset, self.wire_y_offset, self.wire_z_offset, self.wire_yaw_offset, self.origin_position, self.origin_quat)

        self.curr_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.curr_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        wire_pos_body, wire_yaw_body = self.wire_transforms.transform_wire_local_to_body(self.curr_pos, self.curr_quat)

        self.prev_err_x = getattr(self, "prev_err_x", 0)
        self.prev_err_y = getattr(self, "prev_err_y", 0)
        self.prev_err_z = getattr(self, "prev_err_z", 0)
        self.prev_err_yaw = getattr(self, "prev_err_yaw", 0)
        self.sum_err_x = getattr(self, "sum_err_x", 0)
        self.sum_err_y = getattr(self, "sum_err_y", 0)
        self.sum_err_z = getattr(self, "sum_err_z", 0)
        self.sum_err_yaw = getattr(self, "sum_err_yaw", 0)

        if self.first_pass is True:
            self.vx = self.p_xy * wire_pos_body[0] 
            self.vy = self.p_xy * wire_pos_body[1]
            self.vz = self.p_z * wire_pos_body[2]
            self.yaw_rate = self.p_yaw * wire_yaw_body
            dt = 0.0
            self.first_pass = False
        else:
            dt = time.perf_counter() - self.prev_timestamp
            self.vx = self.p_xy * wire_pos_body[0] + self.d_xy * (wire_pos_body[0] - self.prev_err_x) / dt + self.i_xy * self.sum_err_x
            self.vy = self.p_xy * wire_pos_body[1] + self.d_xy * (wire_pos_body[1] - self.prev_err_y) / dt + self.i_xy * self.sum_err_y
            self.vz = self.p_z * wire_pos_body[2] + self.d_z * (wire_pos_body[2] - self.prev_err_z) / dt + self.i_z * self.sum_err_z
            self.yaw_rate = self.p_yaw * wire_yaw_body + self.d_yaw * self.wire_transforms.clamp_angle_rad(wire_yaw_body - self.prev_err_yaw) / dt + self.i_yaw * self.sum_err_yaw

        self.prev_timestamp = time.perf_counter()
        self.prev_err_x = wire_pos_body[0]
        self.prev_err_y = wire_pos_body[1]
        self.prev_err_z = wire_pos_body[2]
        self.prev_err_yaw = wire_yaw_body
        self.sum_err_x += wire_pos_body[0] * dt
        self.sum_err_y += wire_pos_body[1] * dt
        self.sum_err_z += wire_pos_body[2] * dt
        self.sum_err_yaw += wire_yaw_body * dt

        vel_target_msg = PositionTarget()
        vel_target_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        vel_target_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW
        vel_target_msg.velocity.x = np.clip(self.vx, -self.max_velocity_xy, self.max_velocity_xy)
        vel_target_msg.velocity.y = np.clip(self.vy, -self.max_velocity_xy, self.max_velocity_xy)
        vel_target_msg.velocity.z = np.clip(self.vz, -self.max_velocity_z, self.max_velocity_z)
        vel_target_msg.yaw_rate = self.yaw_rate

        self.pos_control_publisher.publish(vel_target_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DroneControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
