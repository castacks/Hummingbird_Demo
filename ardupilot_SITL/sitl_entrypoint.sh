#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /root/ardu_ws/install/setup.bash

ros2 launch ardupilot_gz_bringup iris_runway.launch.py
