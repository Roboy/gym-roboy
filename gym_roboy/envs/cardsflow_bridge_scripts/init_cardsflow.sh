#!/usr/bin/env bash
source "${CARDSFLOW_WS}/devel/setup.bash"

roslaunch kindyn robot.launch robot_name:=msj_platform start_controllers:='sphere_axis0 sphere_axis1 sphere_axis2'
