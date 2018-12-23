The `launch_simulation.sh` script launches the CARDSflow simulation with ROS1. 
It also launches the ROS2 bridge to ROS1. 

The scripts requires you to set the following
environment variables.
* `ROS2_WS`: pointing to your ROS2 installation workspace.
* `ROS2_ROBOY_WS`: pointing to your workspace containing `src/roboy_communications`
* `CARDSFLOW_WS`: pointing to the catkin workspace with your CARDSflow installation
