import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PythonExpression, EnvironmentVariable
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.conditions import IfCondition
import datetime


def generate_launch_description():
    # specify which RViz configuration to use based on WIRE_MODE
    rviz_config_path = PythonExpression([
        '"', FindPackageShare('wire_tracking').find('wire_tracking'), '/rviz/wire_tracking.rviz" if "', EnvironmentVariable('WIRE_MODE'), '" == "2" else "',
        FindPackageShare('wire_detection').find('wire_detection'), '/rviz/wire_detection.rviz"'
    ])
        
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
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('RVIZ'), '" == "1"']))
        ),
        # Visual Servoing Node (launch if SERVO is true)
        Node(
            package='servoing',
            executable='servoing_node',
            name='servoing_node',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('SERVO'), '" == "1"']))
        ),
        
        # MAVROS real Node (launch if MAVROS is set and SIMULATION is false)
        IncludeLaunchDescription(
            XMLLaunchDescriptionSource([FindPackageShare('mavros'), '/launch/apm.launch']),
            condition=IfCondition(PythonExpression([
                '"', EnvironmentVariable('MAVROS'), '" == "1" and "', EnvironmentVariable('SIMULATION'), '" == "0"'
            ]))
        ),
        # MAVROS SITL Node (launch if MAVROS == "1" and SIMULATION == "1")
        IncludeLaunchDescription(
            XMLLaunchDescriptionSource([FindPackageShare('mavros'), '/launch/apm.launch']),
            launch_arguments={'fcu_url': 'udp://:14550@'}.items(),
            condition=IfCondition(PythonExpression([
                '"', EnvironmentVariable('MAVROS'), '" == "1" and "', EnvironmentVariable('SIMULATION'), '" == "1"'
            ]))
        ),
        # MAVROS Manager Node (launch if MAVROS == "1")
        TimerAction(
            period=2.0,  # delay in seconds, adjust as needed
            actions=[
                Node(
                    package='mavros_manager',
                    executable='mavros_manager_node',
                    name='mavros_manager_node',
                    condition=IfCondition(PythonExpression(['"', EnvironmentVariable('MAVROS'), '" == "1"']))
                )
            ]
        ),
        # VINS Node (launch if VO is true)
        Node(
            package='vins',
            executable='vins_node',     
            name='vins_node',
            namespace='vins',
            parameters=[{'config_file': FindPackageShare('vins').find('vins') + '/config/zedx/zedx_stereo_config.yaml'}],
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('VO'), '" == "1"'])),
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
            output='screen',
            condition=IfCondition(PythonExpression(['"', EnvironmentVariable('RECORD'), '" == "1"']))
        ),
    ])

    return system_launch
