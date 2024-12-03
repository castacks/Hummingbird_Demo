#!/usr/bin/env python
import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from .wire_detection_logic import WireDetector

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class WireDetectorNode(Node):
    def __init__(self):
        super().__init__('wire_detection_node')
        self.set_params()

        self.wire_detector = WireDetector()

        # Subscribers
        self.rgb_image_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.image_callback, 10)

        # Publishers
        self.visualization_pub = self.create_publisher(Image, self.visualization_pub_topic, 10)

        self.get_logger().info("Wire Detection Node initialized")


    def image_callback(self, rgb_msg):
        try:
            # Convert the ROS image messages to OpenCV images
            bgr = CvBridge().imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        
        debug_image = None
        debug_image = self.detect_lines_and_update(rgb)
        if debug_image is not None:
            self.publish_debug_image(debug_image)
        else:
            self.publish_debug_image(rgb)

    def detect_lines_and_update(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, 50, 150, apertureSize=3)
        dilation_size = 10
        dilation_kernel = np.ones((dilation_size,dilation_size), np.uint8)
        seg_mask = cv2.dilate(seg_mask, dilation_kernel, iterations=1)
        seg_mask = cv2.erode(seg_mask, dilation_kernel, iterations=1)
        debug_img = image.copy()
        if np.any(seg_mask):
            wire_lines, wire_midpoints, avg_yaw = self.wire_detector.detect_wires(seg_mask)
            debug_img = self.draw_wire_lines(debug_img, wire_lines, wire_midpoints)
            return debug_img

    def publish_debug_image(self, image):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        img_processed_msg = Image()
        img_processed_msg.data = image.tobytes()
        img_processed_msg.encoding = 'rgb8'
        img_processed_msg.header = header
        img_processed_msg.height = image.shape[0]
        img_processed_msg.width = image.shape[1]                
        img_processed_msg.step = image.shape[1] * image.shape[2]
        self.visualization_pub.publish(img_processed_msg)    
    
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
        
    def set_params(self):
        try:
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('visualization_pub_topic', rclpy.Parameter.Type.STRING)
            # Access parameters
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.visualization_pub_topic = self.get_parameter('visualization_pub_topic').get_parameter_value().string_value
        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    

def main():
    rclpy.init()
    node = WireDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()
