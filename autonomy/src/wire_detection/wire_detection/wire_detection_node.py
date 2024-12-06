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
        
        self.bridge = CvBridge()
        self.wire_detector = WireDetector()

        # Subscribers
        self.rgb_image_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.image_callback, 1)

        # Publishers
        self.visualization_pub = self.create_publisher(Image, self.visualization_pub_topic, 1)

        # color tracking varibales
        self.color_dict = {}
        self.max_wires_detected = None

        self.get_logger().info("Wire Detection Node initialized")

    def image_callback(self, rgb_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        start_time = self.get_clock().now()
        debug_image = None
        debug_image = self.detect_lines_and_update(rgb)
        if debug_image is not None:
            img_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='rgb8')
            self.visualization_pub.publish(img_msg)    
        else:
            img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            self.visualization_pub.publish(img_msg)
        end_time = self.get_clock().now()
        # self.get_logger().info(f"Time taken to process image: {end_time - start_time}")

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
    
    def draw_wire_lines(self, img, wire_lines, wire_midpoints):
        # colors = self.assign_line_colors(wire_lines, wire_midpoints)
        colors = []
        colors = [(0, 255, 0) for i in range(len(wire_lines))]
        for i, (x, y) in enumerate(wire_midpoints):
            x0, y0, x1, y1 = wire_lines[i]
            cv2.line(img, (x0, y0), (x1, y1), colors[i], 2)
            cv2.circle(img, (int(x), int(y)), 5, colors[i], -1)
        return img

    def assign_line_colors(self, wire_lines, wire_midpoints):
        colors = []
        if self.color_dict == {} and len(wire_lines) > 0:
            for i in range(len(wire_lines)):  
                color = tuple(np.random.randint(0, 256, 3).tolist())
                self.color_dict[color] = wire_midpoints[i]

        used_colors = []
        for i, (x, y) in enumerate(wire_midpoints): 
            # if all colors are used, assign a random color
            if len(used_colors) == len(self.color_dict):
                color = tuple(np.random.randint(0, 256, 3).tolist())
                self.color_dict[color] = (x, y)
                colors.append(color)
                continue
            
            min_dist = float('inf')
            min_color = None
            for color, midpoint in self.color_dict.items():
                if color in used_colors:
                    continue
                dist = np.linalg.norm(np.array([x, y]) - np.array(midpoint))        
                if dist < min_dist:
                    min_dist = dist
                    min_color = color
            used_colors.append(min_color)
            colors.append(min_color)
            self.color_dict[min_color] = (x, y)
        return colors
        
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
