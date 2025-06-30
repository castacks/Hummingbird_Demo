#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import time

from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import Point
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from wire_interfaces.msg import WireDetection, WireDetections

from .wire_detector_platforms import WireDetectorCPU, WireDetectorGPU
import common_utils.viz_utils as vu

from ament_index_python.packages import get_package_share_directory

# ignore future deprecated warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class WireDetectorNode(Node):
    def __init__(self):
        super().__init__('wire_detection_node')
        self.set_params()
        
        self.bridge = CvBridge()
        self.initialized = False

        # Subscribers
        self.rgb_image_sub = Subscriber(self, Image, self.rgb_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.depth_image_sub = Subscriber(self, Image, self.depth_image_sub_topic, qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.img_tss = ApproximateTimeSynchronizer(
            [self.rgb_image_sub, self.depth_image_sub],
            queue_size=1, 
            slop=0.05
        )
        self.img_tss.registerCallback(self.images_callback)

        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_sub_topic, self.camera_info_callback, 1)

        # Fitted Line Publishers
        self.wire_detections_pub = self.create_publisher(WireDetections, self.wire_detections_pub_topic, 1)

        # Publishers
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 1)
        self.depth_pc_viz_pub = self.create_publisher(PointCloud2, self.depth_pc_pub_topic, 1)

        self.detection_2d_pub = self.create_publisher(Image, self.detections_2d_pub_topic, 1)
        self.detections_3d_pub= self.create_publisher(Marker, self.detections_3d_pub_topic, 1)

    def camera_info_callback(self, data):
        if self.initialized:
            return
        
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.camera_vector = np.array([self.fx, self.fy, self.cx, self.cy])
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                    [0, self.fy, self.cy],
                                    [0, 0, 1]])
        if self.use_cpu:
            self.wire_detector = WireDetectorCPU(get_package_share_directory('wire_detection') + '/config/wire_detection_config.yaml', self.camera_matrix)
        else:
            self.wire_detector = WireDetectorGPU(get_package_share_directory('wire_detection') + '/config/wire_detection_config.yaml', self.camera_matrix)

        self.initialized = True
        self.get_logger().info("WireDetectorNode initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)

    def images_callback(self, rgb_msg, depth_msg):
        if not self.initialized:
            self.get_logger().warn("WireDetectorNode not initialized yet. Waiting for camera info...")
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return
        start_time = time.perf_counter()
        fitted_lines, line_inlier_counts, avg_angle, roi_pcs, roi_point_colors, rgb_masked = self.wire_detector.detect_3d_wires(rgb, depth, generate_viz = self.wire_viz_3d)
        end_time = time.perf_counter()
        # self.get_logger().info(f"Time taken for wire detection: {end_time - start_time:.6f} seconds, {1 / (end_time - start_time):.6f} Hz")

        # Create WireDetections message
        wire_detections_msg = WireDetections()
        wire_detections_msg.header = rgb_msg.header
        wire_detections_msg.header.frame_id = "/left_camera"
        if avg_angle is not None:
            wire_detections_msg.avg_angle = float(avg_angle)
        else:
            wire_detections_msg.avg_angle = float('nan')

        if fitted_lines is not None and len(fitted_lines) > 0:
            for line in fitted_lines:
                wire_estimate = WireDetection()
                wire_estimate.start.x = float(line[0][0])
                wire_estimate.start.y = float(line[0][1])
                wire_estimate.start.z = float(line[0][2])
                wire_estimate.end.x = float(line[1][0])
                wire_estimate.end.y = float(line[1][1])
                wire_estimate.end.z = float(line[1][2])
                wire_estimate.midpoint.x = float((line[0][0] + line[1][0]) / 2)
                wire_estimate.midpoint.y = float((line[0][1] + line[1][1]) / 2)
                wire_estimate.midpoint.z = float((line[0][2] + line[1][2]) / 2)
                wire_estimate.scalar_covariance = 0.0
                wire_detections_msg.wire_detections.append(wire_estimate)
        else:
            wire_detections_msg.wire_detections = []

        self.wire_detections_pub.publish(wire_detections_msg)

        # publish a point cloud for the wires
        self.get_logger().info(f"Number of wires detected: {len(fitted_lines)}")

        # Publish the 3D visualizations if the wire visualization is enabled
        if self.wire_viz_3d:
            self.visualize_3d_point_cloud(depth, rgb)
            self.visualize_3d_wires(fitted_lines)

        # Publish the 3d visualizations if the wire visualization is enabled and wire tracking is not enabled
        if self.wire_viz_2d:
            if fitted_lines is not None and len(fitted_lines) > 0:
                rgb_masked = vu.draw_lines_on_image(rgb_masked.copy(), fitted_lines, self.camera_matrix, color=(0, 255, 0), thickness=2)
                masked_msg = self.bridge.cv2_to_imgmsg(rgb_masked, encoding='rgb8')
                self.detection_2d_pub.publish(masked_msg)
            else:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
                self.detection_2d_pub.publish(rgb_msg)  # Publish original image if no wires detected

            depth_viz = vu.create_depth_viz(depth, self.wire_detector.min_depth_clip, self.wire_detector.max_depth_clip)
            depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding='bgr8')
            self.depth_viz_pub.publish(depth_viz_msg)

    def visualize_3d_point_cloud(self, depth, rgb):
        if self.initialized:
            points, colors = self.wire_detector.depth_to_pointcloud(depth, rgb=rgb)

            # Pack RGB as uint32 and view as float32
            r = (colors[:, 0].astype(np.uint32) & 0xFF) << 16
            g = (colors[:, 1].astype(np.uint32) & 0xFF) << 8
            b = (colors[:, 2].astype(np.uint32) & 0xFF)
            rgb_uint32 = r | g | b
            rgb_float32 = rgb_uint32.view(np.float32)

            # Combine points and rgb field
            pc_data = np.hstack([points.astype(np.float32), rgb_float32[:, np.newaxis]])

            # Define PointFields
            fields = [
                PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "/map"  # or "camera_link" or whatever you're using

            # Create the point cloud message
            pc_msg = pc2.create_cloud(header, fields, pc_data)
            pc_msg.height = 1
            pc_msg.width = pc_data.shape[0]
            pc_msg.is_dense = False

            self.depth_pc_viz_pub.publish(pc_msg)

    def visualize_3d_wires(self, fitted_lines):
        if self.wire_viz_3d and self.initialized:
            # Publish wire visualization
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lines"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            line_scale = 0.05
            marker.scale.x = line_scale
            marker.scale.y = line_scale
            marker.scale.z = line_scale

            # Set color (optional: set per-point if desired)
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

            # Add points (each pair forms a line segment)
            if fitted_lines is None or len(fitted_lines) == 0:
                self.detections_3d_pub.publish(marker)
            else:
                for start, end in fitted_lines:
                    start = start.astype(np.float32)
                    end = end.astype(np.float32)
                    # Create Point objects for the start and end points
                    p1 = Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
                    p2 = Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
                    marker.points.append(p1)
                    marker.points.append(p2)
                    self.detections_3d_pub.publish(marker)
        
    def set_params(self):
        # sub topics
        self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
        self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
        self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
        self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
        self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)
        self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value

        # wire pub topics
        self.declare_parameter('wire_detections_pub_topic', rclpy.Parameter.Type.STRING)
        self.wire_detections_pub_topic = self.get_parameter('wire_detections_pub_topic').get_parameter_value().string_value

        # viz pub topics
        self.declare_parameter('detections_2d_pub_topic', rclpy.Parameter.Type.STRING)
        self.detections_2d_pub_topic = self.get_parameter('detections_2d_pub_topic').get_parameter_value().string_value
        self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
        self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
        self.declare_parameter('depth_pc_pub_topic', rclpy.Parameter.Type.STRING)
        self.depth_pc_pub_topic = self.get_parameter('depth_pc_pub_topic').get_parameter_value().string_value
        self.declare_parameter('detections_3d_pub_topic', rclpy.Parameter.Type.STRING)
        self.detections_3d_pub_topic = self.get_parameter('detections_3d_pub_topic').get_parameter_value().string_value

        # general parameters 
        self.declare_parameter('use_cpu', rclpy.Parameter.Type.BOOL)
        self.use_cpu = self.get_parameter('use_cpu').get_parameter_value().bool_value
        wire_viz = bool(os.getenv('WIRE_VIZ', None).lower())
        wire_mode = int(os.getenv('WIRE_MODE', None).lower())
        self.wire_viz_2d = wire_viz and wire_mode == 1
        self.wire_viz_3d = wire_viz 
    
def main():
    rclpy.init()
    node = WireDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()