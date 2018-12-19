import gym
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

from .. import MsjEnv


def our_env_constructor() -> gym.Env:
    return MsjEnv()


def test_ppo1_agent_learn_successfully():
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([our_env_constructor])

    agent = PPO1(MlpPolicy, env, verbose=1)

    agent.learn(total_timesteps=10)

    assert agent is not None
