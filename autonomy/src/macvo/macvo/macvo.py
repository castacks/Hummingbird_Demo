import rclpy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, CameraInfo
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber

from pathlib import Path
from typing import TYPE_CHECKING
from torchvision.transforms.functional import center_crop, resize
import os, sys
import argparse

from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)
if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import SourceDataFrame, MetaInfo
    from src.Utility.Config import load_config
else:
    from Odometry.MACVO import MACVO                
    from DataLoader import SourceDataFrame, MetaInfo
    from Utility.Config import load_config

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
        self.declare_parameter("model_config", rclpy.Parameter.Type.STRING)
        model_config = self.get_parameter("model_config").get_parameter_value().string_value
        self.get_logger().info(f"Loading macvo model from {model_config}, this might take a while...")
        start_time = self.get_clock().now()
        cfg, _ = load_config(Path(model_config))
        self.odometry   = MACVO.from_config(cfg)
        self.odometry.register_on_optimize_finish(self.publish_latest_pose)
        self.odometry.register_on_optimize_finish(self.publish_latest_points)
        self.odometry.register_on_optimize_finish(self.publish_latest_matches)
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

        self.sync_stereo = ApproximateTimeSynchronizer(
            [self.imageL_sub, self.imageR_sub], queue_size=2, slop=0.1
        )
        self.sync_stereo.registerCallback(self.receive_frame)

        # self.rot_correction_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # self.rot_correction_matrix = np.eye(3)

        self.get_logger().info(f"MACVO Node initialized with camera config: {model_config}")

    def get_camera_info(self, msg: CameraInfo) -> None:
        if self.recieved_camera_info == False:
            self.camera_info = msg
            self.image_width  = msg.width
            self.image_height = msg.height
            self.scale_u = float(self.image_width / self.u_dim)
            self.scale_v = float(self.image_height / self.v_dim)
            
            self.recieved_camera_info = True
            self.get_logger().info(f"Camera info received!")

    def publish_latest_pose(self, system: MACVO):
        pose = system.gmap.frames.pose[-1]
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        out_msg = to_stamped_pose(pose, frame, time)

        # Correction for the camera coordinate frame
        # out_msg.pose.position.x, out_msg.pose.position.y, out_msg.pose.position.z = np.dot(self.rot_correction_matrix, np.array([out_msg.pose.position.x, out_msg.pose.position.y, out_msg.pose.position.z]))
        
        self.pose_pipe.publish(out_msg)
   
    def publish_latest_points(self, system: MACVO):
        if self.point_pipe is None: return
        
        latest_frame  = system.gmap.frames[-1]
        latest_points = system.gmap.get_frame_points(latest_frame)
        latest_obs    = system.gmap.get_frame_observes(latest_frame)
        
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        out_msg = to_pointcloud(
            position  = latest_points.position,
            keypoints = latest_obs.pixel_uv,
            frame_id  = frame,
            colors    = latest_points.color,
            time      = time
        )

        # Correction for the camera coordinate frame
        # for pt in out_msg.points:
        #     pt.x, pt.y, pt.z = np.dot(self.rot_correction_matrix, np.array([pt.x, pt.y, pt.z]))

        self.point_pipe.publish(out_msg)
  
    def publish_latest_stereo(self, system: MACVO):
        if self.img_pipes is None: return
        
        source = system.prev_frame
        if source is None: return
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        img = (source.imageL[0].permute(1, 2, 0).numpy() * 255).copy().astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.frame_id = frame
        msg.header.stamp = time

        self.img_pipes.publish(msg)
    
    def publish_latest_matches(self, system: MACVO):
        if self.img_pipes is None: return

        source = system.prev_frame
        if source is None: return
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        latest_frame  = system.gmap.frames[-1]

        # pixels are given from the reduced image size, need to be scaled back to original size
        pixels_uv    = system.gmap.get_frame_observes(latest_frame).pixel_uv.int().numpy()
        img = (source.imageL[0].permute(1, 2, 0).numpy() * 255).copy().astype(np.uint8)
        if pixels_uv.size > 0:
            for i in range(pixels_uv.shape[0]):
                x, y = pixels_uv[i]
                img = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.frame_id = frame
            msg.header.stamp = time
            self.img_pipes.publish(msg)

    def receive_frame(self, msg_L: Image, msg_R: Image) -> None:
        if self.frame is None or self.bridge is None or self.recieved_camera_info is False:
            self.get_logger().error("MACVO Node not initialized yet, skipping frame")
            return
        
        if self.frame_idx == 0:
            self.get_logger().info(f"Inferencing first frame with scale ({self.scale_u}, {self.scale_v}), please wait...")
            first_frame_time = self.get_clock().now()
        self.prev_time = self.time
        self.time = msg_L.header.stamp
        imageL = self.bridge.imgmsg_to_cv2(msg_L, desired_encoding="passthrough")
        imageR = self.bridge.imgmsg_to_cv2(msg_R, desired_encoding="passthrough")

        camera_fx, camera_fy = self.camera_info.k[0], self.camera_info.k[4]
        camera_cx, camera_cy = self.camera_info.k[2], self.camera_info.k[5]
        meta = MetaInfo(
            idx=self.frame_idx,
            baseline=self.baseline,
            width=self.camera_info.width,
            height=self.camera_info.height,
            K=torch.tensor([[camera_fx, 0., camera_cx],
                            [0., camera_fy, camera_cy],
                            [0., 0., 1.]]).float())
        
        frame = SourceDataFrame(
                meta=meta,
                imageL=torch.tensor(imageL)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imageR=torch.tensor(imageR)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imu=None,
                gtFlow=None, gtDepth=None, gtPose=None, flowMask=None
            ).resize_image(scale_u=self.scale_u, scale_v=self.scale_v)
        
        start_time = self.get_clock().now()
        self.odometry.run(frame)
        end_time = self.get_clock().now()   
        frame_time_diff = (end_time - start_time).nanoseconds / 1e9

        if self.frame_idx == 0:
            time_diff = (end_time - first_frame_time).nanoseconds / 1e9
            self.get_logger().info(f"First frame processed in {time_diff:.2f} seconds.")
        else:
            self.get_logger().info(f"Frame {self.frame_idx} processed in {frame_time_diff}")
            
        self.frame_idx += 1

def main():
    rclpy.init()
    node = MACVONode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
