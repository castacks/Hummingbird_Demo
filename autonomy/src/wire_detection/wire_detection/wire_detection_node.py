#!/usr/bin/env python
import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from common_utils.wire_detection import WireDetector, create_depth_viz

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireDetectorNode(Node):
    def __init__(self):
        super().__init__('wire_detection_node')
        self.set_params()
        
        self.bridge = CvBridge()
        self.wire_detector = WireDetector(threshold=self.line_threshold, expansion_size=self.expansion_size)

        # Subscribers
        self.rgb_image_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.image_callback, 1)
        self.depth_image_sub = self.create_subscription(Image, self.depth_image_sub_topic, self.depth_callback, 1)

        # Publishers
        self.wire_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 1)
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 1)
        self.seg_mask_viz_pub = self.create_publisher(Image, self.seg_mask_pub_topic, 1)

        self.get_logger().info("Wire Detection Node initialized")

    def image_callback(self, rgb_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        debug_image, seg_mask = self.detect_lines_and_update(rgb)
        self.seg_mask_viz_pub.publish(self.bridge.cv2_to_imgmsg(seg_mask, encoding='mono8'))
        if debug_image is not None:
            img_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='rgb8')
            self.wire_viz_pub.publish(img_msg)    
        else:
            img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            self.wire_viz_pub.publish(img_msg)

    def detect_lines_and_update(self, image):
        seg_mask = self.wire_detector.create_seg_mask(image)
        if np.any(seg_mask):
            wire_lines, wire_midpoints, avg_yaw = self.wire_detector.detect_wires(seg_mask)
            self.get_logger().info(f"Num wires detected: {len(wire_midpoints)}")
            debug_img = self.draw_wire_lines(image, wire_lines, wire_midpoints)

            return debug_img, seg_mask
    
    def draw_wire_lines(self, img, wire_lines, wire_midpoints):
        # colors = self.assign_line_colors(wire_lines, wire_midpoints)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        for i, (x, y) in enumerate(wire_midpoints):
            x0, y0, x1, y1 = wire_lines[i]
            cv2.line(img, (x0, y0), (x1, y1), green, 2)
            cv2.circle(img, (int(x), int(y)), 5, blue, -1)
        return img
    
    def depth_callback(self, depth_msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        depth_viz = create_depth_viz(depth)
        img_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding='rgb8')
        self.depth_viz_pub.publish(img_msg)
        
    def set_params(self):
        try:
            #wire detection params
            self.declare_parameter('line_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('expansion_size', rclpy.Parameter.Type.INTEGER)

            # sub pub topics
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('wire_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('seg_mask_pub_topic', rclpy.Parameter.Type.STRING)

            # Access parameters
            self.line_threshold = self.get_parameter('line_threshold').get_parameter_value().integer_value
            self.expansion_size = self.get_parameter('expansion_size').get_parameter_value().integer_value

            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value
            self.wire_viz_pub_topic = self.get_parameter('wire_viz_pub_topic').get_parameter_value().string_value
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.seg_mask_pub_topic = self.get_parameter('seg_mask_pub_topic').get_parameter_value().string_value
        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    
def main():
    rclpy.init()
    node = WireDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()
