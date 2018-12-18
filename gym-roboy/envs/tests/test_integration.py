import numpy as np

from .. import MsjEnv, MsjROSBridgeProxy

env = MsjEnv(MsjROSBridgeProxy())
RANDOM_ACTION = [0.01, 0.01, 0.01, 0.015, 0.01, 0.02, 0.02, 0.02]
ZERO_ACTION = np.zeros(len(RANDOM_ACTION))

def test_msj_env_step():
    obs, reward, done, _ = env.step(RANDOM_ACTION)
    #print(type(done), done)
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
    _, reward, done, _ = env.step( ZERO_ACTION )

    assert done is True
    max_reward = env.reward_range[1]
    assert np.isclose(reward, max_reward)


def test_msj_env_reaching_worst_angle_delivers_lowest_reward():
    goal_joint_angle = np.array([np.pi]*3)
    current_joint_angle = -goal_joint_angle
    expected_reward = -np.linalg.norm(current_joint_angle - goal_joint_angle, ord=2)

    assert np.isclose(env.reward_range[0], expected_reward)

def test_render_does_nothing():
    env.render()
