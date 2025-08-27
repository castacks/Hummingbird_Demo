import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():

   return LaunchDescription([
      Node(
         package='image2rtsp',
         executable='image2rtsp',
         name='image2rtsp',
         parameters=[FindPackageShare('image2rtsp').find('image2rtsp') + '/config/wire_rtsp_config.yaml'],
      )
   ])