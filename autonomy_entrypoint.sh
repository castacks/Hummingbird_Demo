#!/bin/bash
set -e

source /root/dependencies_ws/install/setup.bash

if [ ! -d '/root/ros2_ws/install' ]; then
  source ~/.bash_profile
  colcon build --symlink-install --parallel-workers "$(nproc)" --cmake-args -Wno-dev
else
  echo "Already built"
fi

source /root/ros2_ws/install/setup.bash

# start a seperate process for a ros2 bag record command
if [[ "$RECORD" == "1" ]]; then
  echo "Starting bag recording..."
  ros2 run bag_recorder bag_record_node \
  --ros-args \
  -p cfg_path:=/root/ros2_ws/install/bag_recorder/share/bag_recorder/config/record_topics.yaml \
  -p output_dir:=/root/data_collection/ \
  -p mcap_qos_dir:=/root/ros2_ws/install/bag_recorder/share/bag_recorder/config &
  BAG_PID=$!
fi

cleanup() {
  echo "Shutting down..."
  if [ -n "$BAG_PID" ] && kill -0 "$BAG_PID" 2>/dev/null; then
    echo "Stopping ros2 bag recording..."
    kill -SIGINT "$BAG_PID"
    wait "$BAG_PID"
  fi
}
trap cleanup SIGINT SIGTERM EXIT

# This is the critical part: run ros2 launch as the main process
exec ros2 launch system_bringup bringup.launch.py