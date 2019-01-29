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


@pytest.mark.skip(reason="Needs to update to current 'new_goal' mechanism")
def test_msj_env_new_goal_is_different_and_feasible(msj_env):
    for _ in range(10):
        old_goal = msj_env._goal_joint_angle
        msj_env._set_new_goal()
        assert not np.allclose(old_goal, msj_env._goal_joint_angle)
        assert all(-msj_env._max_joint_angle <= msj_env._goal_joint_angle)
        assert all(msj_env._goal_joint_angle <= msj_env._max_joint_angle)


def test_msj_env_reaching_goal_angle_delivers_maximum_reward(msj_env):
    obs = msj_env.reset()
    current_joint_angle = obs[0:3]
    print("ok")
    msj_env._set_new_goal(goal_joint_angle=current_joint_angle)
    print("ok")
    zero_action = np.zeros(len(msj_env.action_space.low))
    _, reward, done, _ = msj_env.step(zero_action)
    print("ok")

    assert done is True
    max_reward = msj_env.reward_range[1]
    assert np.isclose(reward, max_reward)


@pytest.mark.skip(reason="Do we still have a lowest possible reward?")
def test_msj_env_reaching_worst_angle_delivers_lowest_reward(msj_env):
    goal_joint_angle = np.array([np.pi]*3)
    current_joint_angle = -goal_joint_angle
    expected_reward = -np.linalg.norm(current_joint_angle - goal_joint_angle, ord=2)

    assert np.isclose(msj_env.reward_range[0], expected_reward)


def test_msj_env_render_does_nothing(msj_env):
    msj_env.render()
