#!/usr/bin/env bash

env_variables_are_set() {
    : "${ROS2_WS:?Environment variable 'ROS2_WS' is not set}"
    : "${ROS2_ROBOY_WS:?Environment variable 'ROS2_ROBOY_WS' is not set}"
    : "${CARDSFLOW_WS:?Environment variable 'CARDSFLOW_WS' is not set}"
}

env_variables_are_set

bash "./init_cardsflow.sh" &
sleep 3s
bash "./init_bridge.sh"
