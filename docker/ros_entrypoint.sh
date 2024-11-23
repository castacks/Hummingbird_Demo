#!/bin/bash
set -e

# setup ros2 environment
source "/opt/ros/$ROS_DISTRO/install/setup.bash"
source "/root/ros2_ws/install/local_setup.bash"

# Welcome information
echo "ZED ROS2 Docker Image"
echo "---------------------"
echo 'ROS distro: ' $ROS_DISTRO
echo 'DDS middleware: ' $RMW_IMPLEMENTATION
echo 'ROS 2 Workspaces:' $COLCON_PREFIX_PATH
echo 'ROS 2 Domain ID:' $ROS_DOMAIN_ID
echo 'Machine IPs:' $ROS_IP
echo "---"  
echo 'Available ZED packages:'
if ros2 pkg list | grep zed; then
    echo "ZED packages found."
else
    echo "No ZED packages available or 'ros2' command failed."
fi
echo "---------------------"    
exec "$@"
