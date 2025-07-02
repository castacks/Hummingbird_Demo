import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PythonExpression, EnvironmentVariable
from launch.conditions import IfCondition

def generate_launch_description():
    system_launch = LaunchDescription([
        # Wire tracking node (launch if WIRE_MODE is set to 2)
        Node(
            package='wire_tracking',
            executable='wire_tracking_node',
            name='wire_tracking_node',
            prefix=['xterm -e gdb -ex run --args'],  # For interactive GDB in a separate window/GUI
            output='screen',
            parameters=[FindPackageShare('common_utils').find('common_utils') + '/interface_config.yaml'],
        ),
    ])
    return system_launch