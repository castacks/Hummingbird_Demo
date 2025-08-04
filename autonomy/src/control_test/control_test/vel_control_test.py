#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget
import time
from geometry_msgs.msg import TwistStamped, PoseStamped
import numpy as np

class DroneControlNode(Node):
    def __init__(self):
        super().__init__('drone_control_node')

        # Get parameters
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('velocity', 0.5)
        self.declare_parameter('side_length', 10.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('yaw_rate', 0.0)

        self.control_rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value
        self.side_length = self.get_parameter('side_length').get_parameter_value().double_value
        self.yaw = self.get_parameter('yaw').get_parameter_value().double_value
        self.yaw_rate = self.get_parameter('yaw_rate').get_parameter_value().double_value

        # MAVROS service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.pos_control_publisher = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        self.position_subscriber = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.position_callback,
            10
        )

        # Start mission
        self.set_guided_mode_and_control()

    def position_callback(self, msg):
        self.get_logger().info(f"Current Position: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def call_service(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f'Waiting for {client.srv_name} service...')
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_guided_mode_and_control(self):
        self.get_logger().info("Setting mode to GUIDED...")
        mode_request = SetMode.Request()
        mode_request.custom_mode = "GUIDED"
        mode_response = self.call_service(self.set_mode_client, mode_request)
        if not mode_response or not mode_response.mode_sent:
            self.get_logger().error("Failed to set GUIDED mode")
            return

        time.sleep(5.0) # Wait for mode change

        self.start_publishing_control()

    def start_publishing_control(self):
        self.get_logger().info("Starting control publishing...")
        if rclpy.ok():
            self.execute_side()

    def execute_side(self):
        start_time = time.time()
        side_time = self.side_length / self.velocity
        while rclpy.ok() and (time.time() - start_time < side_time):
            pos_msg = PositionTarget()
            pos_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
            pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
            pos_msg.velocity.x = self.velocity
            pos_msg.velocity.y = 0.0
            pos_msg.velocity.z = 0.0
            pos_msg.yaw = self.yaw
            pos_msg.yaw_rate = self.yaw_rate
            self.pos_control_publisher.publish(pos_msg)
            time.sleep(1.0 / self.control_rate_hz)
        
        final_pos_msg = PositionTarget()
        final_pos_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        final_pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
        final_pos_msg.velocity.x = 0.0
        final_pos_msg.velocity.y = 0.0
        final_pos_msg.velocity.z = 0.0
        final_pos_msg.yaw = self.yaw
        final_pos_msg.yaw_rate = 0.0
        self.pos_control_publisher.publish(final_pos_msg)
        self.get_logger().info("Final position published, stopping control.")

def main(args=None):
    rclpy.init(args=args)
    node = DroneControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
