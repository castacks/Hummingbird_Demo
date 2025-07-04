#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from mavros_msgs.msg import PositionTarget, AttitudeTarget
import time
from tf_transformations import quaternion_from_euler
import numpy as np

POS = 1
POS_RAW = 2
POS_VEL_RAW = 3
ATT = 4

class TimeValueSeries:
    def __init__(self):
        self.time_value_pairs = []

    def add(self, time, value):
        self.time_value_pairs.append((float(time), value))
        self.time_value_pairs = sorted(self.time_value_pairs, key=lambda x:x[0])

    def get_value_at_time(self, desired_time):
        dt = desired_time % self.time_value_pairs[-1][0]
        for index, tv in enumerate(zip(reversed(self.time_value_pairs))):
            time, value = tv[0]
            if dt >= time:
                return value
        return None

    def get_times(self):
        return list(map(lambda x:x[0], self.time_value_pairs))

    def get_values(self):
        return list(map(lambda x:x[1], self.time_value_pairs))


class DroneControlNode(Node):
    def __init__(self):
        super().__init__('drone_control_node')

        # Get parameters
        self.takeoff_wait_time = 10.
        self.takeoff_height = 5.
        self.control_rate_hz = 3.0 # hz

        # MAVROS service clients
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        self.control_publisher = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        # Start mission
        self.set_guided_mode_and_takeoff()

    def call_service(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f'Waiting for {client.srv_name} service...')
        
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_guided_mode_and_takeoff(self):
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

        self.start_publishing_control()

    def start_publishing_control(self):
        self.get_logger().info("Starting control publishing...")
        while rclpy.ok():
            msg = PositionTarget()
            msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            msg.type_mask = PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW = 1024

            msg.velocity.x = 1.0
            msg.velocity.y = 0.
            msg.velocity.z = 0.
            msg.yaw_rate = 0.0

            self.control_publisher.publish(msg)
            time.sleep(1./self.control_rate_hz)


def main(args=None):
    rclpy.init(args=args)
    node = DroneControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
