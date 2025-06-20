import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PythonExpression, EnvironmentVariable
from launch.conditions import IfCondition


def generate_launch_description():
    # Get current date for rosbag naming
    date = str(os.popen('date +%Y-%m-%d_%H:%M:%S').read().strip())
    system_launch = LaunchDescription([
        # Wire tracking node (launch if DETECTION is true)
        Node(
            package='wire_tracking',
            executable='wire_tracking_node',
            name='wire_tracking_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(PythonExpression([EnvironmentVariable('TRACKING')]))
        ),
        # Wire detection node (launch if WIRE_SYS is true and USE_TRACKING is false)
        Node(
            package='wire_detection',
            executable='wire_detection_node',
            name='wire_detection_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(PythonExpression([EnvironmentVariable('DETECTION')]))
        ),
        # Wire Tracking RViz (launch if RVIZ is true and USE_TRACKING is true)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', FindPackageShare('wire_tracking').find('wire_tracking') + '/rviz/wire_tracking.rviz',
                '--ros-args', '--log-level', 'WARN'
            ],
            condition=IfCondition(PythonExpression([EnvironmentVariable('RVIZ'), ' and ', EnvironmentVariable('TRACKING')]))
        ),
        # Wire Detection RViz (launch if RVIZ is true and USE_TRACKING is false)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', FindPackageShare('wire_detection').find('wire_detection') + '/rviz/wire_detection.rviz',
                '--ros-args', '--log-level', 'WARN'
            ],
            condition=IfCondition(PythonExpression([EnvironmentVariable('RVIZ'), ' and not ', EnvironmentVariable('TRACKING'), ' and ', EnvironmentVariable('DETECTION')]))
        ),

        Node(
            package='visual_servoing',
            executable='visual_servoing_node',
            name='visual_servoing_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(EnvironmentVariable('SERVO'))
        ),
        Node(
            package='mavros',
            executable='mavros_node',
            name='mavros_node',
            output='screen',
            parameters=[{
                'fcu_url': '/dev/ttyUSB0:57600',  # Update if using a different port or baudrate
                'gcs_url': 'udp://@',             # Optional — for forwarding MAVLink to GCS
            }],
            condition=IfCondition(EnvironmentVariable('MAVROS'))
        ),
        # VINS Node (launch if VO is true)
        Node(
            package='vins',
            executable='vins_node',
            name='vins_node',
            namespace='vins',
            parameters=[{'config_file': FindPackageShare('vins').find('vins') + '/config/zedx/zedx_stereo_config.yaml'}],
            condition=IfCondition(EnvironmentVariable('VO'))
        ),

        # Rosbag recording process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-d', '60', '-a',
                 '-o', ['/root/data_collection/', 'wire_tracking_', date]],
            output='log',
            condition=IfCondition(EnvironmentVariable('RECORD'))
        ),

    ])

    return system_launch