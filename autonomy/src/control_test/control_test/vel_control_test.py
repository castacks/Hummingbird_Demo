#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, LivelinessPolicy
import time
from geometry_msgs.msg import PoseStamped
import numpy as np

class DroneControlNode(Node):
    def __init__(self):
        super().__init__('velocity_control_node')

        # Get parameters
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('velocity', 0.5)
        self.declare_parameter('travel_distance', 10.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('yaw_rate', 0.0)
        self.declare_parameter('takeoff_height', 2.0)  # Height to take off to
        self.declare_parameter('takeoff_wait_time', 10.0)  # Time to wait after takeoff before starting control

        self.control_rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value
        self.travel_distance = self.get_parameter('travel_distance').get_parameter_value().double_value
        self.yaw = self.get_parameter('yaw').get_parameter_value().double_value
        self.yaw_rate = self.get_parameter('yaw_rate').get_parameter_value().double_value
        self.takeoff_wait_time = self.get_parameter('takeoff_wait_time').get_parameter_value().double_value
        self.takeoff_height = self.get_parameter('takeoff_height').get_parameter_value().double_value

        self.get_logger().info(f"using control_rate_hz: {self.control_rate_hz}")
        self.get_logger().info(f"using velocity: {self.velocity}")
        self.get_logger().info(f"using travel_distance: {self.travel_distance}")
        self.get_logger().info(f"using yaw: {self.yaw}")
        self.get_logger().info(f"using yaw_rate: {self.yaw_rate}")
        self.get_logger().info(f"using takeoff_height: {self.takeoff_height}")
        self.get_logger().info(f"using takeoff_wait_time: {self.takeoff_wait_time}")      

        # MAVROS service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.pos_control_publisher = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,   # UNKNOWN not valid → fallback to KEEP_LAST
            depth=5,                           # Required with KEEP_LAST, choose any small positive integer
            durability=DurabilityPolicy.VOLATILE,
            liveliness=LivelinessPolicy.AUTOMATIC,
        )

        self.position_subscriber = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.position_callback,
            qos_profile
        )

        # Start mission
        self.takeoff_and_control()

    def position_callback(self, msg):
        self.get_logger().info(f"Current Position: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def call_service(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f'Waiting for {client.srv_name} service...')
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def takeoff_and_control(self):
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
            self.get_logger().info(f"Waiting {self.takeoff_wait_time} seconds before control...")
            time.sleep(self.takeoff_wait_time)

        self.get_logger().info("Starting control publishing...")
        self.execute_side()

    def execute_side(self):
        start_time = time.time()
        side_time = self.travel_distance / self.velocity
        while rclpy.ok() and (time.time() - start_time < side_time):
            pos_msg = PositionTarget()
            pos_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
            if self.yaw_rate == 0.0 and self.yaw != 0.0:
                pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW_RATE
                self.get_logger().info("Using yaw control")
            elif self.yaw_rate != 0.0 and self.yaw == 0.0:
                pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW
                self.get_logger().info("Using yaw rate control")
            else:
                pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
                self.get_logger().info("Using no yaw control")
            pos_msg.velocity.x = self.velocity
            pos_msg.velocity.y = 0.0 # Move in the x direction
            pos_msg.velocity.z = 0.0
            pos_msg.yaw = self.yaw
            pos_msg.yaw_rate = self.yaw_rate
            self.pos_control_publisher.publish(pos_msg)
            time.sleep(1.0 / self.control_rate_hz)
        
        final_pos_msg = PositionTarget()
        final_pos_msg.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        final_pos_msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
        final_pos_msg.velocity.x = 0.0
        final_pos_msg.velocity.y = 0.0
        final_pos_msg.velocity.z = 0.0
        final_pos_msg.yaw = 0.0
        final_pos_msg.yaw_rate = 0.0
        self.pos_control_publisher.publish(final_pos_msg)
        self.get_logger().info("Final position published, stopping control.")
        return

def main(args=None):
    rclpy.init(args=args)
    node = DroneControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
