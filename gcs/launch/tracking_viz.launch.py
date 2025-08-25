
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="rqt_image_view",
            output="screen",
            arguments=["/pincer/tracking_2d_viz"]  # change to your image topic
        )
    ])
