import numpy as np
from ..robots import MsjRobot, RobotState

MSJ_ROBOT = MsjRobot()


def test_msj_robot_state_interpolate():
    state1 = MsjRobot.new_random_state()
    state2 = MsjRobot.new_random_state()
    middle_state = RobotState.interpolate(state1, state2)

    assert np.allclose(middle_state.joint_angles, (state1.joint_angles + state2.joint_angles) / 2)
    assert np.allclose(middle_state.joint_vels, (state1.joint_vels + state2.joint_vels) / 2)


def test_msj_robot_state_new_random_zero_angle_state():
    zero_angle_state = MsjRobot.new_random_zero_angle_state()
    assert np.allclose(zero_angle_state.joint_angles, 0)
    assert not np.allclose(zero_angle_state.joint_vels, 0)


def test_msj_robot_state_new_random_zero_vel_state():
    zero_vel_state = MsjRobot.new_random_zero_vel_state()
    assert np.allclose(zero_vel_state.joint_vels, 0)
    assert not np.allclose(zero_vel_state.joint_angles, 0)


def test_robot_new_max_state():
    max_state = MSJ_ROBOT.new_max_state()
    assert np.allclose(max_state.joint_angles, MSJ_ROBOT.get_joint_angles_space().high)
    assert np.allclose(max_state.joint_vels, MSJ_ROBOT.get_joint_vels_space().high)


def test_robot_new_min_state():
    min_state = MSJ_ROBOT.new_min_state()
    assert np.allclose(min_state.joint_angles, MSJ_ROBOT.get_joint_angles_space().low)
    assert np.allclose(min_state.joint_vels, MSJ_ROBOT.get_joint_vels_space().low)


def test_robot_normalize_max_state():
    max_angles = np.ones(MSJ_ROBOT.get_joint_angles_space().shape)
    max_vels = np.ones(MSJ_ROBOT.get_joint_vels_space().shape)

    normed_max_state = MSJ_ROBOT.normalize_state(state=MSJ_ROBOT.new_max_state())
    assert np.allclose(normed_max_state.joint_angles, max_angles)
    assert np.allclose(normed_max_state.joint_vels, max_vels)

    normed_min_state = MSJ_ROBOT.normalize_state(state=MSJ_ROBOT.new_min_state())
    assert np.allclose(normed_min_state.joint_angles, -max_angles)
    assert np.allclose(normed_min_state.joint_vels, -max_vels)
