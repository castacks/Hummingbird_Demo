#!/bin/bash
set -e

source /root/dependencies_ws/install/setup.bash
source /root/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=68
ros2 launch zed_wrapper zed_dual_camera.launch.py pose_cam_serial:=${POSE_CAM_SERIAL} wire_cam_serial:=${WIRE_CAM_SERIAL}