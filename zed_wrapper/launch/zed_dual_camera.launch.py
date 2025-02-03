import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Assume the child launch files are in the same directory as this file
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Paths to the child launch files
    launch_file_path = os.path.join(current_dir, 'zed_camera.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
                'pose_cam_serial',
                default_value=TextSubstitution(text='41607017'),
                description='the camera serial number used for pose estimation'),
        DeclareLaunchArgument(
                'wire_cam_serial',
                default_value=TextSubstitution(text='47859404'),
                description='the camera serial number used for wire tracking'),
        DeclareLaunchArgument(
            'log_level',
            default_value=TextSubstitution(text='warn'),
            description='Logging level (e.g., debug, info, warn, error, fatal)'),

        # Pose camera launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file_path),
            launch_arguments={
                'camera_name': 'pose_cam',
                'camera_model': 'zedx',
                'serial_number': LaunchConfiguration('pose_cam_serial'),
            }.items()
        ),
 
        # Wire camera launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file_path),
            launch_arguments={
                'camera_name': 'wire_cam',
                'camera_model': 'zedx',
                'serial_number': LaunchConfiguration('wire_cam_serial'),
            }.items()
        ),
    ])