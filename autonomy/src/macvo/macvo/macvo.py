import rclpy
import torch
import pypose as pp
import numpy as np
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
import common_utils.coord_transforms as ct


from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import logging

from .DispartyPublisher import DisparityPublisher
from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, src_path)
if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from src.Utility.Config import load_config
else:
    import DataLoader
    from Odometry.MACVO import MACVO
    from DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from Utility.Config import load_config

PACKAGE_NAME = "macvo"

    
class MACVONode(Node):
    def __init__(self):
        super().__init__("macvo_node")
        self.frame_id = 0  # Frame ID
        self.init_time = None  # ROS2 time stamp
        self.get_logger().set_level(logging.INFO)
        self.declared_parameters = set()

        self.get_logger().info("Initializing MACVO Node ...")

        self.bridge = None
        self.time = None
        self.baseline = None
        self.odometry = None
        self.coord_frame = "map"
        self.camera_info = None 
        self.recieved_camera_info = False
        self.meta = None
        self.bridge = CvBridge()

        # Declare publishers ----------------
        self.declare_parameter("pose_pub_topic", rclpy.Parameter.Type.STRING)
        pose_topic = self.get_parameter("pose_pub_topic").get_parameter_value().string_value
        self.pose_pub  = self.create_publisher(PoseStamped, pose_topic, qos_profile=1)

        self.declare_parameter("map_pub_topic", rclpy.Parameter.Type.STRING)
        map_topic = self.get_parameter("map_pub_topic").get_parameter_value().string_value
        self.map_pub = self.create_publisher(PointCloud, map_topic, qos_profile=1)

        self.declare_parameter("feature_img_pub_topic", rclpy.Parameter.Type.STRING)
        feature_img_topic = self.get_parameter("feature_img_pub_topic").get_parameter_value().string_value
        self.feature_pub = self.create_publisher(Image, feature_img_topic, qos_profile=1)
        
        # Wait for camera info to be recieved   
        self.declare_parameter("camera_info_sub_topic", rclpy.Parameter.Type.STRING)
        camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_sub_topic, self.get_camera_info, qos_profile=1)
        
        # Load the MACVO model ------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("Using device: " + str(device) + ", number of devices: " + str(torch.cuda.device_count()))
        start_time = self.get_clock().now()
        
        self.declare_parameter("model_config", rclpy.Parameter.Type.STRING)
        macvo_config_path = self.get_parameter("model_config").get_parameter_value().string_value
        self.get_logger().info(
            f"Loading macvo model from {macvo_config_path}, this might take a while..."
        )
        cfg, _ = load_config(Path(macvo_config_path))

        original_cwd = os.getcwd()
        try:
            os.chdir(get_package_share_directory(PACKAGE_NAME))
            self.odometry = MACVO[StereoFrame].from_config(cfg)
            self.odometry.register_on_optimize_finish(self.publish_data)
        finally:
            os.chdir(original_cwd)

        end_time = self.get_clock().now()
        time_diff = (end_time - start_time).nanoseconds / 1e9
        self.get_logger().info(f"MACVO Model loaded in {time_diff:.2f} seconds. Initializing MACVO node ...")

        self.declare_parameter("inference_dim_u", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("inference_dim_v", rclpy.Parameter.Type.INTEGER)
        self.u_dim = self.get_parameter("inference_dim_u").get_parameter_value().integer_value
        self.v_dim = self.get_parameter("inference_dim_v").get_parameter_value().integer_value

        self.declare_parameter("camera_baseline", rclpy.Parameter.Type.DOUBLE)
        self.baseline = self.get_parameter("camera_baseline").get_parameter_value().double_value

        self.declare_parameter("imageL_sub_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("imageR_sub_topic", rclpy.Parameter.Type.STRING)
        imageL_topic = self.get_parameter("imageL_sub_topic").get_parameter_value().string_value
        imageR_topic = self.get_parameter("imageR_sub_topic").get_parameter_value().string_value
        self.imageL_sub = Subscriber(self, Image, imageL_topic, qos_profile=1)
        self.imageR_sub = Subscriber(self, Image, imageR_topic, qos_profile=1)

        self.sync_stereo = ApproximateTimeSynchronizer(
            [self.imageL_sub, self.imageR_sub], queue_size=2, slop=0.1
        )
        self.sync_stereo.registerCallback(self.receive_stereo)

        # 180 degree rotation around x-axis
        x_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # 90 degree rotation around z-axis
        z_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        self.correction_matrix = np.eye(4)
        self.correction_matrix[:3, :3] = np.dot(x_rot, z_rot)

        self.get_logger().info(f"MACVO Node initialized successfully!")

    def get_camera_info(self, msg: CameraInfo) -> None:
        if self.recieved_camera_info == False:
            self.camera_info = msg
            self.image_width  = msg.width
            self.image_height = msg.height

            self.recieved_camera_info = True
            self.get_logger().info(f"Camera info received! Image size: {self.image_width}x{self.image_height}")

    def publish_data(self, system: MACVO):
        # Latest pose
        pose = pp.SE3(system.graph.frames.data["pose"][-1])
        time_ns = int(system.graph.frames.data["time_ns"][-1].item())

        time = Time()
        time.sec = (time_ns // 1_000_000_000) + self.init_time.sec
        time.nanosec = (time_ns % 1_000_000_000) + self.init_time.nanosec

        pose_msg = to_stamped_pose(pose, self.coord_frame, time)
        transform = ct.pose_to_homogeneous(pose_msg.pose)
        transform = np.dot(self.correction_matrix, transform)
        pose_msg.pose = ct.homogeneous_to_pose(transform)

        # Latest map
        if system.mapping:
            points = system.graph.get_frame2map(system.graph.frames[-2:-1])
        else:
            points = system.graph.get_match2point(
                system.graph.get_frame2match(system.graph.frames[-1:])
            )

        map_pc_msg = to_pointcloud(
            position=points.data["pos_Tw"],
            keypoints=None,
            frame_id=self.coord_frame,
            colors=points.data["color"],
            time=time,
        )

        self.pose_pub.publish(pose_msg)
        self.map_pub.publish(map_pc_msg)

    def publish_latest_matches(self, system: MACVO):
        pixels_uv = system.graph.get_frame2match(system.graph.frames[-1:])
        img = (system.prev_keyframe.imageL[0].permute(1, 2, 0).numpy() * 255).copy().astype(np.uint8)
        if pixels_uv.size > 0:
            for i in range(pixels_uv.shape[0]):
                x, y = pixels_uv[i]
                img = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.frame_id = self.coord_frame
            msg.header.stamp = self.time
            self.feature_pub.publish(msg)

    @staticmethod
    def time_to_ns(time: Time) -> int:
        return int(time.sec * 1e9) + time.nanosec

    def receive_stereo(self, msg_imageL: Image, msg_imageR: Image) -> None:
        if not self.recieved_camera_info:
            self.get_logger().info("Waiting for camera info ...")
            return
        imageL, timestamp = from_image(msg_imageL), msg_imageL.header.stamp
        imageR = from_image(msg_imageR)
        if self.init_time is None:
            self.init_time = timestamp
        elapsed = int(self.time_to_ns(timestamp) - self.time_to_ns(self.init_time))
        # Instantiate a frame and scale to the desired height & width
        # if self.disparity_publisher is not None:
        #     self.disparity_publisher.curr_timestamp = timestamp

        stereo_frame = SmartResizeFrame(
            {
                "height": self.u_dim,
                "width": self.v_dim,
                "interp": "bilinear",
            }
        )(
            StereoFrame(
                idx=[self.frame_id],
                time_ns=[elapsed],
                stereo=StereoData(
                    T_BS=pp.identity_SE3(1, dtype=torch.float),
                    K=torch.tensor(
                        [
                            [
                                [self.camera_info.k[0], 0.0, self.camera_info.k[2]],
                                [0.0, self.camera_info.k[4], self.camera_info.k[5]],
                                [0.0, 0.0, 1.0],
                            ]
                        ],
                        dtype=torch.float,
                    ),
                    baseline=torch.tensor([self.baseline], dtype=torch.float),
                    time_ns=[elapsed],
                    height=imageL.shape[0],
                    width=imageL.shape[1],
                    imageL=torch.tensor(imageL)[..., :3]
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0,
                    imageR=torch.tensor(imageR)[..., :3]
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0,
                ),
            )
        )
        start_time = self.get_clock().now()
        self.odometry.run(stereo_frame)
        end_time = self.get_clock().now()
        time_diff = (end_time - start_time).nanoseconds / 1e9
        self.get_logger().info(f"Frame id {self.frame_id} processed in {time_diff} s")

        # Pose-processing
        self.frame_id += 1

    def destroy_node(self):
        self.odometry.terminate()


def main():
    rclpy.init()
    node = MACVONode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
