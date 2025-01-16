import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
   config = os.path.join(
      get_package_share_directory('image2rtsp'),
      'config',
      'parameters.yaml'
      )

   wire_cam_params = {
      'topic':'/wire_cam/depth_vis',
      'port':'8556'
   }
   pose_cam_params = {
      'topic':'/pose_cam/zed_node/left/image_rect_color',
      'port':'8554'
   }
   return LaunchDescription([
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='image2rtsp',
         parameters=[config, wire_cam_params]
      ),
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='image2rtsp',
         parameters=[config, pose_cam_params]
      )
   ])