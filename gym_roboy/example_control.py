import os

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

from .envs import MsjEnv


MOUNT_DIR = "/root/develDeepAndReinforced/tensorboard_out/"
MODEL_FILE = os.path.join(MOUNT_DIR, "model")


def main():
    # The algorithms require a vectorized environment to run
    env_constructor = MsjEnv
    env = DummyVecEnv([env_constructor])

    if os.path.isfile(MODEL_FILE):
        agent = PPO1.load(MODEL_FILE)
    else:
        agent = PPO1(MlpPolicy, env, verbose=1, tensorboard_log=MOUNT_DIR)

    while True:
        agent.learn(total_timesteps=1000000)
        agent.save(MODEL_FILE)


if __name__ == '__main__':
    main()
