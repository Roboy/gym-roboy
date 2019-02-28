import random
from itertools import combinations

import numpy as np
import pytest

from ..simulations import RosSimulationClient
from ..robots import MsjRobot

SIMULATION_CLIENT = RosSimulationClient(robot=MsjRobot())


@pytest.mark.integration
def test_simulation_client_reset():
    """reset function sets all joint angles to zero in simulation"""

    new_robot_state = SIMULATION_CLIENT.forward_reset_command()
    assert np.allclose([0, 0, 0], new_robot_state.joint_angles)
    assert np.allclose([0, 0, 0], new_robot_state.joint_vels)


@pytest.mark.integration
def test_simulation_client_step():
    """calling the step function changes the robot state"""

    random_action = [0.01, 0.01, 0.01, 0.015, 0.01, 0.02, 0.02, 0.02]
    initial_robot_state = SIMULATION_CLIENT.forward_step_command(random_action)

    new_robot_state = SIMULATION_CLIENT.forward_step_command(random_action)

    assert not np.allclose(initial_robot_state.joint_angles, new_robot_state.joint_angles)
    assert not np.allclose(initial_robot_state.joint_vels, new_robot_state.joint_vels)


@pytest.mark.integration
def test_simulation_client_read_state():
    """calling the read state function doesn't change the robot state"""

    initial_robot_state = SIMULATION_CLIENT.read_state()
    new_robot_state = SIMULATION_CLIENT.read_state()

    assert np.allclose(initial_robot_state.joint_angles, new_robot_state.joint_angles)
    assert np.allclose(initial_robot_state.joint_vels, new_robot_state.joint_vels)


@pytest.mark.integration
def test_simulation_client_get_new_goal_joint_angles_results_are_different():
    different_joint_angles = [SIMULATION_CLIENT.get_new_goal_joint_angles() for _ in range(5)]
    for joint_angle1, joint_angle2 in combinations(different_joint_angles, 2):
        assert not np.allclose(joint_angle1, joint_angle2)


@pytest.mark.integration
def test_simulation_client_stepping_on_the_boundary_does_not_reset():
    random.seed(0)
    actions = MsjRobot.get_action_space()
    strong_action = [float(random.choice(i)) for i in zip(actions.high, actions.low)]
    SIMULATION_CLIENT.forward_reset_command = lambda: pytest.fail("should not call this")

    for _ in range(1000):
        robot_state = SIMULATION_CLIENT.forward_step_command(action=strong_action)
        if not robot_state.is_feasible:
            break

    assert not robot_state.is_feasible
    robot_state = SIMULATION_CLIENT.forward_step_command(action=strong_action)
    assert not robot_state.is_feasible


@pytest.mark.integration
def test_simulation_client_load_test():
    sim_client = RosSimulationClient(robot=MsjRobot())
    for idx in range(100):
        sim_client.forward_reset_command()
        sim_client.get_new_goal_joint_angles()
        print("completed" + str(idx))
