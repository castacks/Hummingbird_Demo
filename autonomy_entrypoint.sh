#!/bin/bash
set -e

source /root/dependencies_ws/install/setup.bash

if [ ! -d '/root/ros2_ws/install' ]; then
  source ~/.bash_profile
  colcon build --symlink-install --parallel-workers $(nproc) --cmake-args -Wno-dev
else
  echo "Already built"
fi

source /root/ros2_ws/install/setup.bash

# This is the critical part: run ros2 launch as the main process
exec ros2 launch system_bringup bringup.launch.py