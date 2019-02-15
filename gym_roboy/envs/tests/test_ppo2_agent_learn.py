from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from .test_msj_env import msj_env  # pytest fixture import


def test_ppo1_agent_learn_successfully(msj_env):
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: msj_env])

    agent = PPO2("MlpPolicy", env=env)

    agent.learn(total_timesteps=10)

    assert agent is not None
