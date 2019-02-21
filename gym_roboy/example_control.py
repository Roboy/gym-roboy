import os

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from .envs import RoboyEnv, ROSBridgeProxy
from .envs.robots import MsjRobot

TRAINING_STEPS_BETWEEN_BACKUPS = 1000000
RESULTS_DIR = "./training_results"
MODEL_FILE = "./model.pkl"


def main():
    # The algorithms require a vectorized environment to run
    env_constructor = lambda: RoboyEnv(ros_proxy=ROSBridgeProxy(robot=MsjRobot()))
    env = DummyVecEnv([env_constructor])

    if os.path.isfile(MODEL_FILE):
        agent = PPO2.load(MODEL_FILE, tensorboard_log=RESULTS_DIR)
        agent.set_env(env)
    else:
        agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR, ent_coef=0.1)

    while True:
        agent.learn(total_timesteps=TRAINING_STEPS_BETWEEN_BACKUPS)
        agent.save(MODEL_FILE)


if __name__ == '__main__':
    main()
