from typing import Sequence

import numpy as np
from itertools import combinations

import pytest

from .. import MsjEnv, MockMsjROSProxy, MsjRobotState, MsjROSBridgeProxy

constructors = [
    lambda: MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=True),
    lambda: MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False),
    pytest.param(lambda: MsjEnv(ros_proxy=MsjROSBridgeProxy(), joint_vel_penalty=False), marks=pytest.mark.integration)
]


@pytest.fixture(
    params=constructors,
    ids=["unit-test-default", "unit-test-no-joint-vel-penalty", "integration"]
)
def msj_env(request) -> MsjEnv:
    return request.param()


def test_msj_env_step(msj_env):
    msj_env.reset()
    obs, reward, done, _ = msj_env.step(msj_env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool), str(type(done))


def test_msj_env_reset(msj_env):
    all_obs = [msj_env.reset() for _ in range(5)]

    for obs in all_obs:
        assert np.allclose(obs[:6], 0)
        assert isinstance(obs, np.ndarray)

    for obs1, obs2 in combinations(all_obs, 2):
        assert np.allclose(obs1, obs2)


def test_msj_env_new_goal_is_different_and_feasible(msj_env: MsjEnv):
    for _ in range(3):
        old_goal = np.array(msj_env._goal_joint_angle)
        msj_env._set_new_goal()
        assert not np.allclose(old_goal, msj_env._goal_joint_angle)
        assert np.all(-msj_env._JOINT_ANGLE_BOUNDS <= msj_env._goal_joint_angle)
        assert np.all(msj_env._goal_joint_angle <= msj_env._JOINT_ANGLE_BOUNDS)


def test_msj_env_reaching_goal_angle_delivers_maximum_reward(msj_env: MsjEnv):
    obs = msj_env.reset()
    current_joint_angle = obs[0:3]
    msj_env._set_new_goal(goal_joint_angle=current_joint_angle)
    zero_action = np.zeros(len(msj_env.action_space.low))
    _, reward, done, _ = msj_env.step(zero_action)

    max_reward = msj_env.reward_range[1]
    assert np.isclose(reward, max_reward)


def test_msj_env_reaching_goal_joint_angle_but_moving_returns_done_equals_false(msj_env: MsjEnv):
    obs = msj_env.reset()
    current_joint_angle = obs[0:3]
    msj_env._set_new_goal(goal_joint_angle=current_joint_angle)

    msj_env._last_state.joint_vel = msj_env._JOINT_VEL_BOUNDS

    assert not msj_env._did_complete_successfully(current_state=msj_env._last_state)


def test_msj_env_joint_vel_penalty_affects_worst_possible_reward():
    env = MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False)
    largest_distance = np.linalg.norm(2 * np.ones_like(MsjEnv._JOINT_ANGLE_BOUNDS))
    worst_possible_reward_from_angles = -np.exp(largest_distance) - abs(MsjEnv._PENALTY_FOR_TOUCHING_BOUNDARY)
    assert np.isclose(env.reward_range[0], worst_possible_reward_from_angles)

    env = MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=True)
    assert env.reward_range[0] < worst_possible_reward_from_angles


def test_msj_env_reward_is_lower_with_joint_vel_penalty():
    new_goal_state = MsjRobotState.new_random_state()
    new_goal_state.joint_vel = np.zeros_like(new_goal_state.joint_vel)
    some_action = MsjEnv.action_space.sample()

    ros_proxy = MockMsjROSProxy()
    ros_proxy.forward_step_command = lambda a: new_goal_state
    env = MsjEnv(ros_proxy=ros_proxy, joint_vel_penalty=False)
    env.reset()
    env._set_new_goal(goal_joint_angle=new_goal_state.joint_angle)
    _, reward_with_no_joint_vel_penalty, _, _ = env.step(action=some_action)

    env = MsjEnv(ros_proxy=ros_proxy, joint_vel_penalty=True)
    env.reset()
    env._set_new_goal(goal_joint_angle=new_goal_state.joint_angle)
    _, reward_with_joint_vel_penalty, _, _ = env.step(action=some_action)

    assert reward_with_no_joint_vel_penalty > reward_with_joint_vel_penalty


def test_msj_env_render_does_nothing(msj_env):
    msj_env.render()


@pytest.mark.parametrize(
    "env",
    [MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=True),
     MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False)],
    ids=["with joint_vel penalty", "no joint_vel penalty"]
)
def test_msj_env_reward_monotonously_improves_during_approach(env: MsjEnv):
    """
    For any random starting state, if we approach the goal at every
    step, the reward should strictly improve. This is like computing
    the numerical gradient dr/dq and checking that it is positive as
    the state approaches the goal state.

    This tests catches mistakes like reaching optimality by not moving.
    """
    np.random.seed(0)
    env.seed(0)

    num_starting_states = 40  # roughly covers many points in the space
    num_steps = 7  # distance to goal halves at every step

    goal_state = MsjRobotState.new_random_zero_vel_state()

    starting_states = [MsjRobotState.new_random_state() for _ in range(num_starting_states)]

    # special cases
    state_with_improvable_angles = MsjRobotState.new_random_zero_vel_state()
    starting_states.append(state_with_improvable_angles)
    if env._joint_vel_penalty:
        state_with_improvable_vels = MsjRobotState.new_random_zero_angle_state()
        starting_states.append(state_with_improvable_vels)

    for current_state in starting_states:
        sequence_of_rewards = []
        for _ in range(num_steps):
            reward = env.compute_reward(current_state=current_state, goal_state=goal_state)
            sequence_of_rewards.append(reward)
            current_state = MsjRobotState.interpolate(current_state, goal_state)  # cut distance by half
        assert _strictly_increasing(sequence_of_rewards)


def _strictly_increasing(sequence: Sequence[float]):
    return all(x < y for x, y in zip(sequence, sequence[1:]))
