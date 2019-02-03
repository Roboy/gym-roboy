import gym
import time
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

from .envs import MsjEnv, MsjROSBridgeProxy


def our_env_constructor() -> gym.Env:
    return MsjEnv(MsjROSBridgeProxy())


def main():
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([our_env_constructor])

    agent = PPO1(MlpPolicy, env, verbose=1)

    timesteps = 1000000
    agent.learn(total_timesteps=timesteps)

    agent_name = "agent_" + str(timesteps) + ".pkl"
    agent_path = "/root/DeepAndReinforced/gym-roboy/gym_roboy/" + agent_name

    agent.save(agent_path)
    #agent = PPO1.load(agent_path)

    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            print("step reward:", reward)
            time.sleep(0.01)
        print("####################\nreached goal!\n####################")


if __name__ == '__main__':
    main()
