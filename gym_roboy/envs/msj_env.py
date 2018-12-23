import numpy as np

import gym
from gym import spaces
from .ros_proxy import MsjROSProxy, MockMsjROSProxy, MsjRobotState


class MsjEnv(gym.GoalEnv):
    reward_range = (-10.88279628, 0)  # max l2 distance given action bounds (-pi, pi) and action dim (3 right now).

    def __init__(self, ros_proxy: MsjROSProxy=MockMsjROSProxy(), seed: int = None):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        
        self._max_joint_angle = np.pi
        self._max_tendon_speed = 0.02  # cm/s
        self._set_new_goal()

        self.action_space = spaces.Box(
            low=-self._max_tendon_speed,
            high=self._max_tendon_speed,
            shape=(self._ros_proxy.DIM_ACTION,)
            , dtype='float32'
        )
        self._l2_distance_for_success = self._l2_distance(
            self.action_space.low, self.action_space.high) / 100  # 100 seems reasonable

        # 3 * DIM_JOINT_ANGLE from the observation = DIM_current_joint_velocity + DIM_current_joints + DIM_goal_joints
        self.observation_space = spaces.Box(-self._max_joint_angle, self._max_joint_angle,
                                            shape=(3*self._ros_proxy.DIM_JOINT_ANGLE,), dtype='float32')

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high).tolist()
        new_state = self._ros_proxy.forward_step_command(action)
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(new_state.joint_angle, self._goal_joint_angle, info)
        done = self._did_reach_goal(actual_joint_angle=new_state.joint_angle)
        return obs, reward, done, info

    def _make_obs(self, robot_state: MsjRobotState):
        return np.concatenate([
            robot_state.joint_angle,
            robot_state.joint_vel,
            self._goal_joint_angle
        ])

    def reset(self):
        self._ros_proxy.forward_reset_command()
        return self._make_obs(robot_state=self._ros_proxy.read_state())

    def render(self, mode='human'):
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        current_joint_angle = achieved_goal
        return -self._l2_distance(current_joint_angle, desired_goal)

    @staticmethod
    def _l2_distance(joint_angle1, joint_angle2):
        return np.linalg.norm(np.subtract(joint_angle1, joint_angle2), ord=2)

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        if goal_joint_angle is not None:
            self._goal_joint_angle = goal_joint_angle
            return
        new_joint_angle = np.random.random(self._ros_proxy.DIM_JOINT_ANGLE)
        self._goal_joint_angle = np.clip(new_joint_angle, -self._max_joint_angle, self._max_joint_angle)

    def _did_reach_goal(self, actual_joint_angle) -> bool:
        l2_distance = self._l2_distance(actual_joint_angle, self._goal_joint_angle)
        return bool(l2_distance < self._l2_distance_for_success) # bool for comparison to a numpy bool
