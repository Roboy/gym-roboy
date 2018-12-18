import numpy as np

from .. import MsjEnv

env = MsjEnv()
RANDOM_ACTION = env._action_space.sample()


def test_msj_env_step():
    obs, reward, done, _ = env.step(RANDOM_ACTION)
    assert "observation" in obs
    assert isinstance(obs["observation"], np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_msj_reset():
    obs = env.reset()
    assert "observation" in obs
    assert isinstance(obs["observation"], np.ndarray)


def test_msj_new_goal_is_different_and_feasible():
    for _ in range(10):
        old_goal = env._goal_joint_angle
        env._set_new_goal()
        assert not np.allclose(old_goal, env._goal_joint_angle)
        assert all(-env._max_joint_angle <= env._goal_joint_angle)
        assert all(env._goal_joint_angle <= env._max_joint_angle)


def test_msj_reaching_goal_angle_delivers_maximum_reward():
    obs = env.reset()
    current_joint_angle = obs["achieved_goal"]
    env._set_new_goal(goal_joint_angle=current_joint_angle)
    _, reward, done, _ = env.step(RANDOM_ACTION)

    assert done is True
    max_reward = env.reward_range[1]
    assert np.isclose(reward, max_reward)


def test_msj_reaching_worst_angle_delivers_minimum_rewards():
    obs = env.reset()
    opposite_of_current_joint_angle = -obs["achieved_goal"]
    env._set_new_goal(goal_joint_angle=opposite_of_current_joint_angle)
    _, reward, done, _ = env.step(RANDOM_ACTION)

    assert done is False
    minimum_reward = env.reward_range[0]
    assert np.isclose(reward, minimum_reward)


def test_cosine_similarity():
    vector = np.random.random(5)
    precision = 0.0001
    assert 1 - env._cosine_similarity(vector, vector) < precision
    assert -1 - env._cosine_similarity(vector, -vector) < precision


def test_render_does_nothing():
    env.render()
