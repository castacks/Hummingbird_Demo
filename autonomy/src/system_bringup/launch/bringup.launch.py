import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PythonExpression, EnvironmentVariable
from launch.conditions import IfCondition


def generate_launch_description():
    # Get current date for rosbag naming
    date = str(os.popen('date +%Y-%m-%d').read().strip())
    system_launch = LaunchDescription([
        # Wire tracking node (launch if WIRE_NODE is true)
        Node(
            package='wire_tracking',
            executable='wire_tracking_node',
            name='wire_tracking_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/wire_tracking_config.yaml'],
            condition=IfCondition(PythonExpression([
                EnvironmentVariable('WIRE_SYS'), ' and ', EnvironmentVariable('USE_TRACKING')
            ]))

        ),
        # Wire detection node (launch if WIRE_SYS is true and USE_TRACKING is false)
        Node(
            package='wire_detection',
            executable='wire_detection_node',
            name='wire_detection_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/wire_tracking_config.yaml'],
            condition=IfCondition(PythonExpression([
                EnvironmentVariable('WIRE_SYS'), ' and not ', EnvironmentVariable('USE_TRACKING')
            ]))

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
            condition=IfCondition(PythonExpression([EnvironmentVariable('RVIZ'), ' and ', EnvironmentVariable('USE_TRACKING')]))
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
            condition=IfCondition(PythonExpression([EnvironmentVariable('RVIZ'), ' and not ', EnvironmentVariable('USE_TRACKING')]))
        ),


        # Rosbag recording process
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-d', '60',
                 '-o', ['/root/data_collection/', 'wire_tracking_', date], '/wire_cam/zed_node/left/image_rect_color',
                '/wire_cam/zed_node/right/image_rect_color', '/wire_cam/zed_node/left/camera_info',
                '/wire_cam/zed_node/right/camera_info', '/wire_cam/zed_node/depth/depth_registered',
                '/pose_cam/zed_node/left/image_rect_color', '/pose_cam/zed_node/right/image_rect_color', 
                '/pose_cam/zed_node/left/camera_info', '/pose_cam/zed_node/right/camera_info'],
            output='log',
            condition=IfCondition(EnvironmentVariable('RECORD'))
        ),

        # VINS Node (launch if VO is true)
        Node(
            package='vins',
            executable='vins_node',
            name='vins_node',
            parameters=[{'config_file': FindPackageShare('vins').find('vins') + '/config/zedx/zedx_stereo_config.yaml'}],
            condition=IfCondition(EnvironmentVariable('VO'))
        ),
    ])

    return system_launch