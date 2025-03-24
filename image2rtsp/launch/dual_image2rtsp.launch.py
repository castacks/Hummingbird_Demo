import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
   detection_config = os.path.join(
      get_package_share_directory('image2rtsp'),
      'config',
      'detection_config.yaml'
      )
      
   depth_config = os.path.join(
      get_package_share_directory('image2rtsp'),
      'config',
      'depth_config.yaml'
      )
      
   return LaunchDescription([
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='detection_rtsp',
         parameters=[detection_config]
      ),
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='depth_rtsp',
         parameters=[depth_config]
      )
   ])