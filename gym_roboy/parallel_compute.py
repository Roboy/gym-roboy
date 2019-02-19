import os
import sys

import gym
import gym_roboy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from .envs import MsjEnv, MsjROSBridgeProxy
import time

RESULTS_DIR = "./training_results"
#assert not os.path.isdir(RESULTS_DIR), "Folder '{}' already exists".format(RESULTS_DIR)
MODEL_FILE = "./model.pkl"

def check_time():
    last_time = None
    def callback(input1, input2):
        nonlocal last_time
        if last_time is not None:
            end = time.time()
            print(str(end - last_time))
            last_time = end
        else:
            last_time = time.time()
    return callback

def setup_constructor(rank, seed=0):
    def our_env_constructor() -> gym.Env:
        environment = MsjEnv(MsjROSBridgeProxy(process_idx= rank))
        environment.seed(seed + rank)
        return environment

    return our_env_constructor


num_cpu = int(sys.argv[1])
env = SubprocVecEnv([setup_constructor(i + 1) for i in range(num_cpu)])
agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR)

while True:
    agent.learn(total_timesteps=100000, callback=check_time())
    agent.save(MODEL_FILE)


