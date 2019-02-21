import sys

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from .envs import RoboyEnv, ROSBridgeProxy
from .envs.robots import MsjRobot

RESULTS_DIR = "./training_results"
MODEL_FILE = "./model.pkl"


def setup_constructor(rank, seed=0):
    def our_env_constructor() -> gym.Env:
        environment = RoboyEnv(ROSBridgeProxy(process_idx= rank, robot=MsjRobot()))
        environment.seed(seed + rank)
        return environment

    return our_env_constructor


num_cpu = int(sys.argv[1])
env = SubprocVecEnv([setup_constructor(i + 1) for i in range(num_cpu)])
agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR)

while True:
    agent.learn(total_timesteps=1000000)
    agent.save(MODEL_FILE)


