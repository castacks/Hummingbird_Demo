#!/usr/bin/env python
import rclpy
import rclpy.clock
from rclpy.node import Node
import numpy as np
import cv2
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import Point
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from wire_detection.wire_detector_platform import WireDetectorCPU, WireDetectorGPU
import common_utils.visualization_utils as wd

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

        # Publishers
        self.wire_2d_viz_pub = self.create_publisher(Image, self.wire_2d_viz_pub_topic, 1)
        self.depth_viz_pub = self.create_publisher(Image, self.depth_viz_pub_topic, 1)
        self.depth_pc_viz_pub = self.create_publisher(PointCloud2, self.depth_pc_pub_topic, 1)
        self.wire_3d_viz_pub= self.create_publisher(Marker, self.wire_3d_viz_pub_topic, 1)

    def camera_info_callback(self, data):
        if self.received_camera_info:
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
            self.wire_detector = WireDetectorCPU(get_package_share_directory('wire_detection') + '/config/wire_detection_config.yaml', camera_matrix=self.camera_matrix)
        else:
            self.wire_detector = WireDetectorGPU(get_package_share_directory('wire_detection') + '/config/wire_detection_config.yaml', camera_matrix=self.camera_matrix)

        self.initialized = True
        self.get_logger().info("WireDetectorNode initialized with camera info.")
        self.destroy_subscription(self.camera_info_sub)

    def images_callback(self, rgb_msg, depth_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            rclpy.logerr("CvBridge Error: {0}".format(e))
            return

        fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.wire_detector.detect_3d_wires(self, rgb, depth, generate_viz = self.vizualize_wires)
        
        # publish a point cloud for the wires
        self.visualize_3d_detection(depth, fitted_lines)

        masked_msg = self.bridge.cv2_to_imgmsg(rgb_masked, encoding='rgb8')
        self.wire_2d_viz_pub.publish(masked_msg)

    def visualize_3d_detection(self, depth, rgb, fitted_lines):
        if self.initialized:
            # Publish the point cloud
            pc_msg = PointCloud2()

            points, colors = self.wire_detector.depth_to_pointcloud(depth, rgb=rgb)
            r = np.array(colors[:, 0], dtype=np.float32)
            g = np.array(colors[:, 1], dtype=np.float32)
            b = np.array(colors[:, 2], dtype=np.float32)
            rgb = (r << 16) | (g << 8) | b

            pc_data = np.hstack([points, rgb[:, np.newaxis].astype(np.float32)])
            pc_data = pc_data.astype(np.float32)
            
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=3)
            ]
            pc_msg = pc2.create_cloud(Header(), fields, pc_data)
            pc_msg.height = 1  # Unordered point cloud
            pc_msg.width = pc_data.shape[0]
            pc_msg.is_dense = False

            self.depth_pc_viz_pub.publish(pc_msg)

            # Publish wire visualization
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lines"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Line width

            # Set color (optional: set per-point if desired)
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

            # Add points (each pair forms a line segment)
            for start, end in fitted_lines:
                p1 = Point(x=start[0], y=start[1], z=start[2])
                p2 = Point(x=end[0], y=end[1], z=end[2])
                marker.points.append(p1)
                marker.points.append(p2)

            self.wire_3d_viz_pub.publish(marker)
        
    def set_params(self):
        try:
            # sub topics
            self.declare_parameter('camera_info_sub_topic', rclpy.Parameter.Type.STRING)
            self.camera_info_sub_topic = self.get_parameter('camera_info_sub_topic').get_parameter_value().string_value
            self.declare_parameter('rgb_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.rgb_image_sub_topic = self.get_parameter('rgb_image_sub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_image_sub_topic', rclpy.Parameter.Type.STRING)
            self.depth_image_sub_topic = self.get_parameter('depth_image_sub_topic').get_parameter_value().string_value

            # pub topics
            self.declare_parameter('wire_2d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_2d_viz_pub_topic = self.get_parameter('wire_2d_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_viz_pub_topic = self.get_parameter('depth_viz_pub_topic').get_parameter_value().string_value
            self.declare_parameter('depth_pc_pub_topic', rclpy.Parameter.Type.STRING)
            self.depth_pc_pub_topic = self.get_parameter('depth_pc_pub_topic').get_parameter_value().string_value
            self.declare_parameter('wire_3d_viz_pub_topic', rclpy.Parameter.Type.STRING)
            self.wire_3d_viz_pub_topic = self.get_parameter('wire_3d_viz_pub_topic').get_parameter_value().string_value

            # general parameters 
            self.declare_parameter('use_cpu', rclpy.Parameter.Type.BOOL)
            self.use_cpu = self.get_parameter('use_cpu').get_parameter_value().bool_value
            self.declare_parameter('vizualize_wires', rclpy.Parameter.Type.BOOL)
            self.vizualize_wires = self.get_parameter('vizualize_wires').get_parameter_value().bool_value

        except Exception as e:
            self.get_logger().info(f"Error in declare_parameters: {e}")
    
def main():
    rclpy.init()
    node = WireDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()