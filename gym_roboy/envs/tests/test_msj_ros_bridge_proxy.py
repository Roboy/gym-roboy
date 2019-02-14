from itertools import combinations

import numpy as np
import pytest

from gym_roboy.envs import MsjEnv, MsjRobotState
from .. import MsjROSBridgeProxy


ros_bridge_proxy = MsjROSBridgeProxy()


@pytest.mark.integration
def test_msj_ros_bridge_proxy_reset():
    """reset function sets all joint angles to zero in simulation"""

    new_robot_state = ros_bridge_proxy.forward_reset_command()
    assert np.allclose([0, 0, 0], new_robot_state.joint_angle)
    assert np.allclose([0, 0, 0], new_robot_state.joint_vel)


@pytest.mark.integration
def test_msj_ros_bridge_proxy_step():
    """calling the step function changes the robot state"""

    random_action = [0.01, 0.01, 0.01, 0.015, 0.01, 0.02, 0.02, 0.02]
    initial_robot_state = ros_bridge_proxy.forward_step_command(random_action)

    new_robot_state = ros_bridge_proxy.forward_step_command(random_action)

    for x, y in zip(initial_robot_state.joint_angle, new_robot_state.joint_angle):
        assert np.abs(x - y) > 0.00001

    for x, y in zip(initial_robot_state.joint_vel, new_robot_state.joint_vel):
        assert (np.abs(x - y) > 0.00001 or np.abs(x - y) == 0)


@pytest.mark.integration
def test_msj_ros_bridge_proxy_read_state():
    """calling the read state function doesn't change the robot state"""

    initial_robot_state = ros_bridge_proxy.read_state()
    new_robot_state = ros_bridge_proxy.read_state()

    assert np.allclose(initial_robot_state.joint_angle, new_robot_state.joint_angle)
    assert np.allclose(initial_robot_state.joint_vel, new_robot_state.joint_vel)


@pytest.mark.integration
def test_msj_ros_bridge_proxy_get_new_goal_joint_angles_results_are_different():
    different_joint_angles = [ros_bridge_proxy.get_new_goal_joint_angles() for _ in range(5)]
    for joint_angle1, joint_angle2 in combinations(different_joint_angles, 2):
        assert not np.allclose(joint_angle1, joint_angle2)


@pytest.mark.integration
def test_msj_env_stepping_on_the_boundary_does_not_reset():
    strong_action = [MsjEnv._MAX_TENDON_VEL]*4 + [0.0]*(MsjRobotState.DIM_ACTION-4)
    ros_bridge_proxy.forward_reset_command = lambda: pytest.fail("should not call this")

    for _ in range(1000):
        robot_state = ros_bridge_proxy.forward_step_command(action=strong_action)
        if not robot_state.is_feasible:
            break

    assert not robot_state.is_feasible
    robot_state = ros_bridge_proxy.forward_step_command(action=strong_action)
    assert not robot_state.is_feasible
