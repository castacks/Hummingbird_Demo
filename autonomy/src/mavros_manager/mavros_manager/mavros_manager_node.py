#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import MessageInterval

class MavrosManager(Node):
    def __init__(self):
        super().__init__('mavros_manager')
        self.client = self.create_client(MessageInterval, '/mavros/set_message_interval')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /mavros/set_message_interval service...')

        # Example stream rates: [stream_id, rate_hz, on_off]
        streams = [
            (32, 20, True),  # LOCAL_POSITION_NED
            (33, 20, True),  # GLOBAL_POSITION_INT
            (63, 20, True),  # GLOBAL_POSITION_INT_COV
            (64, 10, True),  # LOCAL_POSITION_NED_COV
        ]
        for stream_id, rate, on_off in streams:
            req = MessageInterval.Request()
            req.message_id = stream_id
            req.message_rate = float(rate)
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info(f'Set stream {stream_id} to {rate} Hz: {future.result()}')

        self.get_logger().info('All streams set, shutting down...')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = MavrosManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
