from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    date = str(os.popen('date +%Y-%m-%d').read().strip())
    return LaunchDescription([
        # Declare environment variables
        DeclareLaunchArgument('WIRE_NODE', default_value='false', description='Enable wire tracking node'),
        DeclareLaunchArgument('RVIZ', default_value='false', description='Enable RViz'),
        DeclareLaunchArgument('ROSBAG_PATH', default_value='/tmp/rosbag', description='Path for rosbag recording'),
        DeclareLaunchArgument('LOCAL_VO', default_value='false', description='Enable local VO node'),

        # Wire tracking node
        Node(
            package='wire_tracking',
            executable='wire_tracking_node',
            name='wire_tracking_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/wire_tracking_config.yaml'],
            condition=LaunchConfiguration('WIRE_NODE')
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', FindPackageShare('wire_tracking').find('wire_tracking') + '/rviz/wire_tracking.rviz',
                '--ros-args', '--log-level', 'WARN'
            ],
            condition=LaunchConfiguration('RVIZ')
        ),

        # Rosbag recording
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-d', '60', '-o', 'wire_tracking_' + date],
            output='log'
        ),

        # VINS Node
        Node(
            package='vins',
            executable='vins_node',
            name='vins_node',
            parameters=[{'config_file': FindPackageShare('vins').find('vins') + '/config/zedx/zedx_stereo_config.yaml'}],
            condition=LaunchConfiguration('LOCAL_VO')
        ),
    ])