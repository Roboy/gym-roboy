#!/usr/bin/env bash
echo "Please source ROS2 yourself"
python3 -m pytest --run-integration --disable-warnings -v -x $@
