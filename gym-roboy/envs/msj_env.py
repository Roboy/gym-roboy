import numpy as np

import gym
from gym import spaces
from .ros_proxy import MsjROSProxy, MockMsjROSProxy, MsjRobotState


class MsjEnv(gym.GoalEnv):
    reward_range = (-1.0, 1.0)

    def __init__(self, ros_proxy: MsjROSProxy=MockMsjROSProxy(), seed: int = None):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._min_cosine_similarity_for_success = 0.9
        self._max_joint_angle = np.pi
        self._max_tendon_speed = 0.02  # cm/s
        self._set_new_goal()

        self._action_space = spaces.Box(
            low=-self._max_tendon_speed,
            high=self._max_tendon_speed,
            shape=(self._ros_proxy.DIM_ACTION,)
            , dtype='float32'
        )

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-self._max_joint_angle, self._max_joint_angle, shape=(self._ros_proxy.DIM_JOINT_ANGLE,), dtype='float32'),
            achieved_goal=spaces.Box(-self._max_joint_angle, self._max_joint_angle, shape=(self._ros_proxy.DIM_JOINT_ANGLE,), dtype='float32'),
            observation=spaces.Box(-self._max_joint_angle, self._max_joint_angle, shape=(3*self._ros_proxy.DIM_JOINT_ANGLE,), dtype='float32'),
        ))

    def step(self, action):
        action = np.clip(action, self._action_space.low, self._action_space.high)
        new_state = self._ros_proxy.forward_step_command(action)
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(obs['achieved_goal'], self._goal_joint_angle, info)
        done = self._did_reach_goal(obs['achieved_goal'])
        return obs, reward, done, info

    def _make_obs(self, robot_state: MsjRobotState):
        full_obs = np.concatenate(
            [robot_state.joint_angle, robot_state.joint_vel, self._goal_joint_angle]
        )
        return {
            'observation': full_obs,
            'achieved_goal': robot_state.joint_angle.copy(),
            'desired_goal': self._goal_joint_angle.copy(),
        }

    def reset(self):
        self._ros_proxy.forward_reset_command()
        return self._make_obs(robot_state=self._ros_proxy.read_state())

    def render(self, mode='human'):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        current_joint_angle = achieved_goal
        return self._cosine_similarity(current_joint_angle, self._goal_joint_angle)

    @staticmethod
    def _cosine_similarity(angle1: np.ndarray, angle2: np.ndarray):
        """https://en.wikipedia.org/wiki/Cosine_similarity"""
        norm = np.linalg.norm
        return angle1.dot(angle2) / (norm(angle1, ord=2)*norm(angle2, ord=2))

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        if goal_joint_angle is not None:
            self._goal_joint_angle = goal_joint_angle
            return
        new_joint_angle = np.random.random(self._ros_proxy.DIM_JOINT_ANGLE)
        self._goal_joint_angle = np.clip(new_joint_angle, -self._max_joint_angle, self._max_joint_angle)

    def _did_reach_goal(self, actual_joint_angle) -> bool:
        cos_similarity = self._cosine_similarity(
            actual_joint_angle, self._goal_joint_angle)
        return bool(cos_similarity > self._min_cosine_similarity_for_success)
