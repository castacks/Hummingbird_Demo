import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
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
                default_value=TextSubstitution(text='0'),
                description='the camera serial number used for pose estimation'),
        DeclareLaunchArgument(
                'wire_cam_serial',
                default_value=TextSubstitution(text='0'),
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
               'config_path': os.path.join(get_package_share_directory('zed_wrapper'),'config','pose_common.yaml'),
            }.items()
        ),

        # IMPORTATNT: Launch must be delayed, having issues when both cameras are launched at the same time
        # Wire camera launch file
        TimerAction(
            period=1.0,  # 1 second delay
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(launch_file_path),
                    launch_arguments={
                        'camera_name': 'wire_cam',
                        'camera_model': 'zedx',
                        'serial_number': LaunchConfiguration('wire_cam_serial'),
                        'config_path': os.path.join(get_package_share_directory('zed_wrapper'), 'config', 'wire_common.yaml'),
                    }.items()
                )
            ]
        )
    ])