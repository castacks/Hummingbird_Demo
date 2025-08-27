#!/bin/bash
set -e

source /root/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=68

ros2 launch image2rtsp image2rtsp.launch.py