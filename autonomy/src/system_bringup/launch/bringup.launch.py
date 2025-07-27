import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PythonExpression, EnvironmentVariable
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.conditions import IfCondition


def generate_launch_description():
    # Get current date for rosbag naming
    date = str(os.popen('date +%Y-%m-%d_%H:%M:%S').read().strip())
    # specify which RViz configuration to use based on WIRE_MODE
    rviz_config_path = PythonExpression([
            '"', FindPackageShare('wire_tracking').find('wire_tracking'), '/rviz/wire_tracking.rviz" if "', EnvironmentVariable('WIRE_MODE'), '" == "2" else "',
            FindPackageShare('wire_detection').find('wire_detection'), '/rviz/wire_detection.rviz"']),
    system_launch = LaunchDescription([
        # Wire tracking node (launch if WIRE_MODE is set to 2)
        Node(
            package='wire_tracking',
            executable='wire_tracking_node',
            name='wire_tracking_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('WIRE_MODE'), '" == "2"'])),
        ),
        # Wire detection node (launch if WIRE_MODE is set to 1 or 2)
        Node(
            package='wire_detection',
            executable='wire_detection_node',
            name='wire_detection_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('WIRE_MODE'), '" == "1" or "', EnvironmentVariable('WIRE_MODE'), '" == "2"'])),
        ),
        # RViz Node (launch if RVIZ is true)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d', rviz_config_path,
                '--ros-args', '--log-level', 'WARN'
            ],
            condition=IfCondition(PythonExpression([EnvironmentVariable('RVIZ')]))
        ),
        # Visual Servoing Node (launch if SERVO is true)
        Node(
            package='visual_servoing',
            executable='visual_servoing_node',
            name='visual_servoing_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(EnvironmentVariable('SERVO'))
        ),
        
        # MAVROS Node (launch if MAVROS is set)
        IncludeLaunchDescription(XMLLaunchDescriptionSource([FindPackageShare('mavros'),'/launch/apm.launch']),
        condition=IfCondition(EnvironmentVariable('MAVROS'))),

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
        Node(
            package='bag_recorder',
            executable='bag_record_node',
            name='bag_recorder',
            parameters=[
                {'cfg_path': FindPackageShare('bag_recorder').find('bag_recorder') + '/config/record_topics.yaml'},
                {'output_dir': '/root/data_collection/'},
                {'mcap_qos_dir': FindPackageShare('bag_recorder').find('bag_recorder') + '/config'}
            ],
            output='screen'
        ),

        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-d', '60',
        #         '-o', f'/root/data_collection/zed_{date}',
        #         '/wire_cam/zed_node/left/image_rect_color', '/wire_cam/zed_node/right/image_rect_color', 
        #         '/wire_cam/zed_node/left/camera_info', '/wire_cam/zed_node/right/camera_info', 
        #         '/wire_cam/zed_node/depth/depth_registered', 
        #         '/pose_cam/zed_node/left/image_rect_color', '/pose_cam/zed_node/right/image_rect_color', 
        #         '/pose_cam/zed_node/left/camera_info', '/pose_cam/zed_node/right/camera_info'],
        #     output='log',
        #     condition=IfCondition(EnvironmentVariable('RECORD'))
        # ),
        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-d', '60',
        #         '-o', f'/root/data_collection/mavros_{date}',
        #          '--regex', '/mavros/.*'],
        #     output='log',
        #     condition=IfCondition(EnvironmentVariable('RECORD'))
        # ),
    ])

    return system_launch
