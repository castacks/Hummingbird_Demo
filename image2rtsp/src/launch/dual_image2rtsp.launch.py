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

   stream1_params = {
      'topic':'/wire_cam/detection_debug',
      'port':'8556'
   }
   stream2_params = {
      'topic':'/wire_cam/depth_debug',
      'port':'8554'
   }
   return LaunchDescription([
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='image2rtsp',
         parameters=[config, stream1_params]
      ),
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='image2rtsp',
         parameters=[config, stream2_params]
      )
   ])