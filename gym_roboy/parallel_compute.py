import os
import sys

import gym
import gym_roboy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2



RESULTS_DIR = "./training_results"
assert not os.path.isdir(RESULTS_DIR), "Folder '{}' already exists".format(RESULTS_DIR)
MODEL_FILE = "./model.pkl"

def setup_constructor(multi_process, rank, seed=0):

    def our_env_constructor() -> gym.Env:
        environment = gym.make("msj-control-v0", multi_process = multi_process, process_idx = rank)
        environment.seed(seed + rank)
        return environment

    return our_env_constructor

num_cpu = int(sys.argv[1])
multi_process = bool(num_cpu > 1)
env = SubprocVecEnv([setup_constructor(multi_process, i) for i in range(num_cpu)])
agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR)

while True:
    agent.learn(total_timesteps=1000000)
    agent.save(MODEL_FILE)


