import numpy as np
from itertools import combinations

import pytest

from .. import MsjEnv, MockMsjROSProxy, MsjRobotState

constructors = [
    lambda: MsjEnv(ros_proxy=MockMsjROSProxy()),
    lambda: MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False),
    pytest.param(lambda: MsjEnv(), marks=pytest.mark.integration)
]


@pytest.fixture(
    params=constructors,
    ids=["unit-test-default", "unit-test-no-joint-vel-penalty", "integration"]
)
def msj_env(request) -> MsjEnv:
    return request.param()


def test_msj_env_step(msj_env):
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

    strong_action = msj_env.action_space.high
    _, _, done, _ = msj_env.step(strong_action)

    assert not done


def test_msj_env_joint_vel_penalty_affects_worst_possible_reward():
    env = MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False)
    worst_possible_reward_from_angles = -np.linalg.norm(2*np.ones_like(MsjEnv._JOINT_ANGLE_BOUNDS))
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
