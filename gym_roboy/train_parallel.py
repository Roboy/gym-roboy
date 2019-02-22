import os
import sys

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from .envs import RoboyEnv, ROSBridgeProxy
from .envs.robots import MsjRobot


is_results_dir_passed = len(sys.argv) is 3
RESULTS_DIR = os.path.abspath(sys.argv[2]) if is_results_dir_passed else "./training_results"
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")
MODEL_FILE = os.path.join(RESULTS_DIR, "model.pkl")
TRAINING_STEPS_BETWEEN_BACKUPS = 1000000


def setup_constructor(rank, seed=0):
    def our_env_constructor() -> gym.Env:
        environment = RoboyEnv(ROSBridgeProxy(process_idx= rank, robot=MsjRobot()))
        environment.seed(seed + rank)
        return environment

    return our_env_constructor


num_cpu = int(sys.argv[1])
env = SubprocVecEnv([setup_constructor(i + 1) for i in range(num_cpu)])
more_exploration = 0.1
agent = PPO2("MlpPolicy", env, tensorboard_log=TENSORBOARD_DIR, ent_coef=more_exploration)

while True:
    agent.learn(total_timesteps=TRAINING_STEPS_BETWEEN_BACKUPS)
    agent.save(MODEL_FILE)
