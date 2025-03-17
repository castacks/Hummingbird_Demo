import rclpy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
import pypose as pp

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, CameraInfo
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time

from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import argparse
import logging

from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)
if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from src.Utility.Config import load_config
    from src.Utility.PrettyPrint import Logger
    from src.Utility.Timer import Timer
else:
    import DataLoader
    from Odometry.MACVO import MACVO                
    from DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from Utility.Config import load_config
    from Utility.PrettyPrint import Logger
    from Utility.Timer import Timer

class MACVONode(Node):
    def __init__(self):
        super().__init__("macvo_node")

        self.bridge = None
        self.time = None
        self.prev_time = None
        self.baseline = None
        self.prev_frame = None
        self.odometry = None
        self.frame = "map"
        self.camera_info = None 
        self.recieved_camera_info = False
        self.meta = None
        self.bridge = CvBridge()

        # Declare publishers ----------------
        self.declare_parameter("pose_pub_topic", rclpy.Parameter.Type.STRING)
        pose_topic = self.get_parameter("pose_pub_topic").get_parameter_value().string_value
        self.pose_pipe  = self.create_publisher(PoseStamped, pose_topic, qos_profile=1)
        
        self.declare_parameter("point_pub_topic", rclpy.Parameter.Type.STRING)
        point_topic = self.get_parameter("point_pub_topic").get_parameter_value().string_value
        self.point_pipe = self.create_publisher(PointCloud, point_topic, qos_profile=1)
        
        self.declare_parameter("img_pub_topic", rclpy.Parameter.Type.STRING)
        img_stream = self.get_parameter("img_pub_topic").get_parameter_value().string_value
        self.img_pipes = self.create_publisher(Image, img_stream, qos_profile=1)
        
        # Wait for camera info to be recieved   
        self.declare_parameter("camera_info_sub_topic", rclpy.Parameter.Type.STRING)
        camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_sub_topic, self.get_camera_info, qos_profile=1)
        
        # Load the MACVO model ------------------------------------
        # check device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info("Using device: " + str(device) + " , number of devices: " + str(torch.cuda.device_count()))

        self.declare_parameter("model_config", rclpy.Parameter.Type.STRING)
        model_config = self.get_parameter("model_config").get_parameter_value().string_value
        self.get_logger().info(f"Loading macvo model from {model_config}, this might take a while...")
        start_time = self.get_clock().now()
        cfg, _ = load_config(Path(model_config))
        self.odometry = MACVO[StereoFrame].from_config(cfg)

        self.odometry.register_on_optimize_finish(self.publish_data)
        self.odometry.register_on_optimize_finish(self.publish_matches)

        end_time = self.get_clock().now()
        time_diff = (end_time - start_time).nanoseconds / 1e9
        self.get_logger().info(f"MACVO Model loaded in {time_diff:.2f} seconds. Initializing MACVO node ...")

        self.frame_idx  = 0

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

        self.frame_fn = SmartResizeFrame({"height": self.u_dim, "width": self.v_dim, "interp": "bilinear"})

        self.sync_stereo = ApproximateTimeSynchronizer(
            [self.imageL_sub, self.imageR_sub], queue_size=2, slop=0.1
        )
        self.sync_stereo.registerCallback(self.receive_frame)

        self.get_logger().info(f"MACVO Node initialized with camera config: {model_config}")

    def get_camera_info(self, msg: CameraInfo) -> None:
        if self.recieved_camera_info == False:
            self.camera_info = msg
            self.image_width  = msg.width
            self.image_height = msg.height

            self.recieved_camera_info = True
            self.get_logger().info(f"Camera info received! Image size: {self.image_width}x{self.image_height}")

    def publish_data(self, system: MACVO):
        # Latest pose
        pose    = pp.SE3(system.graph.frames.data["pose"][-1])
        time_ns = int(system.graph.frames.data["time_ns"][-1].item())
        time = Time()
        time.sec = time_ns // 1_000_000_000
        time.nanosec = time_ns % 1_000_000_000
        pose_msg = to_stamped_pose(pose, self.coord_frame, time)
        
        # Latest map
        if system.mapping:
            points = system.graph.get_frame2map(system.graph.frames[-2:-1])
        else:
            points = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))

        map_pc_msg = to_pointcloud(
            position  = points.data["pos_Tw"],
            keypoints = None,
            frame_id  = self.coord_frame,
            colors    = points.data["color"],
            time      = time,
        )

        self.pose_send.publish(pose_msg)
        self.map_send.publish(map_pc_msg)
   
    def publish_matches(self, system: MACVO):
        if self.img_pipes is None: return

        source = system.prev_frame
        if source is None: return
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        latest_frame  = system.gmap.frames[-1]

        pixels_uv    = system.gmap.get_frame_observes(latest_frame).pixel1_uv.int().numpy()
        img = (source.imageL[0].permute(1, 2, 0).numpy() * 255).copy().astype(np.uint8)
        if pixels_uv.size > 0:
            for i in range(pixels_uv.shape[0]):
                x, y = pixels_uv[i]
                img = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.frame_id = frame
            msg.header.stamp = time
            self.img_pipes.publish(msg)

    def receive_stereo(self, msg_imageL: Image, msg_imageR: Image) -> None:
        self.get_logger().info(f"{self.odometry.graph}")
        imageL, timestamp = from_image(msg_imageL), msg_imageL.header.stamp
        imageR            = from_image(msg_imageR)
        
        # Instantiate a frame and scale to the desired height & width
        stereo_frame = self.frame_fn(StereoFrame(
            idx    =[self.frame_id],
            time_ns=[timestamp.nanosec],
            stereo =StereoData(
                T_BS=pp.identity_SE3(1),
                K   =torch.tensor([[
                    [self.camera.fx, 0.            , self.camera.cx],
                    [0.            , self.camera.fy, self.camera.cy],
                    [0.            , 0.            , 1.            ]
                ]]),
                baseline=torch.tensor([self.camera.bl]),
                time_ns=[timestamp.nanosec],
                height=imageL.shape[0],
                width=imageL.shape[1],
                imageL=torch.tensor(imageL)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imageR=torch.tensor(imageR)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
            )
        ))
        self.odometry.run(stereo_frame)
        
        # Pose-processing
        self.frame_id += 1

def main():
    rclpy.init()
    node = MACVONode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
