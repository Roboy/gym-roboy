import numpy as np
from itertools import combinations

import pytest

from .. import MsjEnv, MockMsjROSProxy

constructors = [
    lambda: MsjEnv(ros_proxy=MockMsjROSProxy()),
    pytest.param(lambda: MsjEnv(), marks=pytest.mark.integration)
]


@pytest.fixture(
    params=constructors,
    ids=["unit-test", "integration"]
)
def msj_env(request) -> MsjEnv:
    return request.param()


def test_msj_env_step(msj_env):
    obs, reward, done, _ = msj_env.step(msj_env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


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


def test_msj_env_reaching_goal_angle_delivers_maximum_reward(msj_env):
    obs = msj_env.reset()
    current_joint_angle = obs[0:3]
    msj_env._set_new_goal(goal_joint_angle=current_joint_angle)
    zero_action = np.zeros(len(msj_env.action_space.low))
    _, reward, done, _ = msj_env.step(zero_action)

    assert done is True
    max_reward = msj_env.reward_range[1]
    assert np.isclose(reward, max_reward)


def test_msj_env_joint_vel_penalty_affects_worst_possible_reward():
    env = MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=False)
    expected_worst_possible_reward = -np.linalg.norm(-MsjEnv._JOINT_ANGLE_BOUNDS - MsjEnv._JOINT_ANGLE_BOUNDS)
    assert np.isclose(env.reward_range[0], expected_worst_possible_reward)

    env = MsjEnv(ros_proxy=MockMsjROSProxy(), joint_vel_penalty=True)
    expected_worst_possible_reward = -np.inf
    assert np.isclose(env.reward_range[0], expected_worst_possible_reward)


def test_msj_env_render_does_nothing(msj_env):
    msj_env.render()
