import os

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

from .envs import MsjEnv


def main():
    # The algorithms require a vectorized environment to run
    env_constructor = MsjEnv
    env = DummyVecEnv([env_constructor])

    agent = PPO1(MlpPolicy, env)


    timesteps = 10
    agent.learn(total_timesteps=timesteps)

    agent_name = "agent_" + str(timesteps) + ".pkl"
    agent_path = "/root/DeepAndReinforced/gym-roboy/" + agent_name

    agent.save(agent_path)


if __name__ == '__main__':
    main()
