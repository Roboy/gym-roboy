#!/usr/bin/env bash
source /opt/ros/kinetic/setup.bash
source "${ROS2_WS}/install/setup.bash"
source "${CARDSFLOW_WS}/devel/setup.bash"
source "${ROS2_ROBOY_WS}/install/setup.bash"

ros2 run ros1_bridge dynamic_bridge
