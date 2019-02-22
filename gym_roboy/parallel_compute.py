import os
import sys

import gym
import gym_roboy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from .envs import MsjEnv, MsjROSBridgeProxy
import numpy as np
import time
RESULTS_DIR = "./training_results"
#assert not os.path.isdir(RESULTS_DIR), "Folder '{}' already exists".format(RESULTS_DIR)
MODEL_FILE = "./model.pkl"


def setup_constructor(rank, seed=0):
    def our_env_constructor() -> gym.Env:
        environment = MsjEnv(MsjROSBridgeProxy(process_idx=rank))
        print(environment._ros_proxy)
        environment.seed(seed + rank)
        return environment

    return our_env_constructor


num_cpu = int(sys.argv[1])

env = SubprocVecEnv([setup_constructor(i + 1) for i in range(num_cpu)])
#env = DummyVecEnv([setup_constructor(i + 1) for i in range(num_cpu)])

#agent = PPO2("MlpPolicy", env, tensorboard_log=RESULTS_DIR)

num_steps = 10000
obs = env.reset()
start_time = time.time()
for _ in range(num_steps):
    action = np.random.random((num_cpu, env.action_space.shape[0]))
    obs, rewards, dones, info = env.step(action)
#agent.learn(num_steps)
total_time_multi = time.time() - start_time

print("Took {:.2f}s for multiprocessed version ".format(total_time_multi))
