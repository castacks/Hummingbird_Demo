#!/usr/bin/env python
import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

import common_utils.wire_detection as wd
import common_utils.coord_transforms as ct

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireDetectorNode(Node):
    def __init__(self):
        super().__init__('wire_detection_node')
        self.set_params()
        
        self.bridge = CvBridge()
        self.wire_detector = wd.WireDetector(threshold=self.line_threshold, 
                                             expansion_size=self.expansion_size, 
                                             low_canny_threshold=self.low_canny_threshold, 
                                             high_canny_threshold=self.high_canny_threshold,
                                             pixel_binning_size=self.pixel_binning_size) 

        # Subscribers
        self.rgb_image_sub = self.create_subscription(Image, self.rgb_image_sub_topic, self.image_callback, 1)
        self.depth_image_sub = self.create_subscription(Image, self.depth_image_sub_topic, self.depth_callback, 1)
        self.camera_info_sub = self.create_subscription(Image, self.depth_image_sub_topic, self.depth_callback, 1)

        # Publishers
        self.wire_viz_pub = self.create_publisher(Image, self.wire_viz_pub_topic, 1)
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 1)
        self.seg_mask_viz_pub = self.create_publisher(Image, self.seg_mask_pub_topic, 1)
        self.pc_viz_pub = self.create_publisher(PointCloud2, self.pc_pub_topic, 1)

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
        
    def camera_info_callback(self, data):
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.camera_vector = np.array([self.fx, self.fy, self.cx, self.cy])
        if self.received_camera_info == False:
            self.get_logger().info("Received Camera Info")
        self.received_camera_info = True
    
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
        if self.received_camera_info == True:
            # make an array of all pixels in an image for N x 2
            pc_msg = PointCloud2()
            pc_msg.header = depth_msg.header
            pc_msg.height = depth.shape[0]
            pc_msg.width = depth.shape[1]

            pixels = np.indices((depth.shape[0], depth.shape[1])).reshape(2, -1).T
            corrsponding_depth = depth.flatten()
            camera_x, camera_y, camera_z = ct.image_to_camera(pixels, corrsponding_depth, self.camera_vector)
            camera_x = camera_x.reshape(depth.shape)
            camera_y = camera_y.reshape(depth.shape)
            camera_z = camera_z.reshape(depth.shape)
            pc_msg.data = np.array([camera_x, camera_y, camera_z], dtype=np.float32).tobytes()
            pc_msg.is_dense = False

            self.pc_viz_pub.publish(pc_msg)

        depth_viz = wd.create_depth_viz(depth)
        img_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding='rgb8')
        self.depth_viz_pub.publish(img_msg)
        
    def set_params(self):
        try:
            #wire detection params
            self.declare_parameter('line_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('expansion_size', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('low_canny_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('high_canny_threshold', rclpy.Parameter.Type.INTEGER)
            self.declare_parameter('pixel_binning_size', rclpy.Parameter.Type.INTEGER)

            # sub topics
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)

            # pub topics
            self.declare_parameter('wire_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('seg_mask_pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('pc_pub_topic', rclpy.Parameter.Type.STRING)

            # Access parameters
            self.line_threshold = self.get_parameter('line_threshold').get_parameter_value().integer_value
            self.expansion_size = self.get_parameter('expansion_size').get_parameter_value().integer_value
            self.low_canny_threshold = self.get_parameter('low_canny_threshold').get_parameter_value().integer_value
            self.high_canny_threshold = self.get_parameter('high_canny_threshold').get_parameter_value().integer_value
            self.pixel_binning_size = self.get_parameter('pixel_binning_size').get_parameter_value().integer_value
            
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value

            self.wire_viz_pub_topic = self.get_parameter('wire_viz_pub_topic').get_parameter_value().string_value
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.seg_mask_pub_topic = self.get_parameter('seg_mask_pub_topic').get_parameter_value().string_value
            self.pc_pub_topic = self.get_parameter('pc_pub_topic').get_parameter_value().string_value

        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    
def main():
    rclpy.init()
    node = WireDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()
