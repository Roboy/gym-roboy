import os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from .envs import MsjEnv, MsjROSBridgeProxy


RESULTS_DIR = "./training_results"
MODEL_FILE = "./model.pkl"


def main():
    # The algorithms require a vectorized environment to run
    env_constructor = lambda: MsjEnv(ros_proxy=MsjROSBridgeProxy())
    env = DummyVecEnv([env_constructor])

    if os.path.isfile(MODEL_FILE):
        agent = PPO2.load(MODEL_FILE, tensorboard_log=RESULTS_DIR)
        agent.set_env(env)
    else:
        agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR)

    while True:
        agent.learn(total_timesteps=1000000)
        agent.save(MODEL_FILE)


if __name__ == '__main__':
    main()
