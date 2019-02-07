#!/usr/bin/env bash
echo "Sourcing ROS2_ROBOY_WS"
source "${ROS2_ROBOY_WS}/install/setup.bash"

python3 -m pytest --run-integration --disable-warnings -v
