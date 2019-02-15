import numpy as np
from .. import MsjRobotState


def test_msj_robot_state_interpolate():
    state1 = MsjRobotState.new_random_state()
    state2 = MsjRobotState.new_random_state()
    middle_state = MsjRobotState.interpolate(state1, state2)

    assert np.allclose(middle_state.joint_angle, (state1.joint_angle + state2.joint_angle) / 2)
    assert np.allclose(middle_state.joint_vel, (state1.joint_vel + state2.joint_vel) / 2)


def test_msj_robot_state_new_random_zero_angle_state():
    zero_angle_state = MsjRobotState.new_random_zero_angle_state()
    assert np.allclose(zero_angle_state.joint_angle, 0)
    assert not np.allclose(zero_angle_state.joint_vel, 0)


def test_msj_robot_state_new_random_zero_vel_state():
    zero_vel_state = MsjRobotState.new_random_zero_vel_state()
    assert np.allclose(zero_vel_state.joint_vel, 0)
    assert not np.allclose(zero_vel_state.joint_angle, 0)
