import os
import sys

# Fail fast before importing stable-baselines
import time

MODEL_FILE = os.path.abspath(sys.argv[1])
assert os.path.isfile(MODEL_FILE), "File not found: '{}'".format(MODEL_FILE)

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from .envs import RoboyEnv
from .envs.simulations import RosSimulationClient
from .envs.robots import MsjRobot


class Logger:
    step = 0

    def log(self, reward):
        if self.step % 1 == 0:
            print("reward:", reward)
        self.step += 1
        time.sleep(0.15)


def main():
    logger = Logger()
    # The algorithms require a vectorized environment to run
    env_constructor = lambda: RoboyEnv(simulation_client=RosSimulationClient(robot=MsjRobot()))
    env = DummyVecEnv([env_constructor])

    agent = PPO2.load(MODEL_FILE)
    agent.set_env(env)

    while True:
        obs = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            logger.log(reward)


if __name__ == '__main__':
    main()
