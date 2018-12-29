#!/usr/bin/env bash

source /opt/ros/kinetic/setup.bash
source "${ROS2_WS}/install/setup.bash"

source "${CARDSFLOW_WS}/devel/setup.bash"
source "${ROS2_ROBOY_WS}/install/setup.bash"

cd ${ROS2_WS}
export MAKEFLAGS=-j1
colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure
