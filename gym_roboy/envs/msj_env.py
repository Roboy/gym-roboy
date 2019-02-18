from typing import Tuple
import numpy as np
import gym
from gym import spaces
from . import MsjROSProxy, MsjRobotState


def _l2_distance(joint_angle1, joint_angle2):
    subtract = np.subtract(joint_angle1, joint_angle2)
    subtract[np.isnan(subtract)] = 0  # np.inf - np.inf returns np.nan, but should be 0
    return np.linalg.norm(subtract, ord=2)


class MsjEnv(gym.GoalEnv):

    _BONUS_FOR_REACHING_GOAL = 10

    def __init__(self, ros_proxy: MsjROSProxy,
                 seed: int = None, joint_vel_penalty: bool = False, is_tendon_vel_dependent_on_distance: bool = False,
                 is_agent_getting_bonus_for_reaching_goal: bool = False):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._joint_vel_penalty = joint_vel_penalty
        self._is_tendon_vel_dependent_on_distance = is_tendon_vel_dependent_on_distance
        self._is_agent_getting_bonus_for_reaching_goal = is_agent_getting_bonus_for_reaching_goal
        self.robot = MsjRobotState

        self._GOAL_JOINT_VEL = np.zeros(self.robot.DIM_JOINT_ANGLE)
        self._MAX_TENDON_VEL_FOR_SUCCESS = self.robot.MAX_TENDON_VEL / 100  # fraction not yet tuned
        self._MAX_DISTANCE_JOINT_ANGLE = _l2_distance(-self.robot.JOINT_ANGLE_BOUNDS, self.robot.JOINT_ANGLE_BOUNDS)
        self._PENALTY_FOR_TOUCHING_BOUNDARY = 1
        self._BONUS_FOR_REACHING_GOAL = 10

        self.reward_range = self._create_reward_range(robot=self.robot)
        self.action_space = MsjRobotState.ACTION_SPACE
        obs_bounds = np.concatenate((self.robot.JOINT_ANGLE_BOUNDS,
                                     self.robot.JOINT_VEL_BOUNDS,
                                     self.robot.JOINT_ANGLE_BOUNDS))
        self.observation_space = spaces.Box(low=-obs_bounds, high=obs_bounds)
        self._set_new_goal()

    def _create_reward_range(self, robot: MsjRobotState) -> Tuple[float, float]:
        some_state = MsjRobotState(joint_angle=robot.JOINT_ANGLE_BOUNDS,
                                   joint_vel=self._GOAL_JOINT_VEL, is_feasible=True)
        corresponding_worst_state = MsjRobotState(joint_angle=-robot.JOINT_ANGLE_BOUNDS,
                                                  joint_vel=-robot.JOINT_VEL_BOUNDS,
                                                  is_feasible=False)
        max_reward = self.compute_reward(current_state=some_state, goal_state=some_state)
        min_reward = self.compute_reward(current_state=corresponding_worst_state, goal_state=some_state)
        return min_reward, max_reward

    def step(self, action):
        assert self.action_space.contains(action)

        if self._is_tendon_vel_dependent_on_distance:
            distance_to_target = _l2_distance(self._last_state.joint_angle, self._goal_state.joint_angle)
            scale_factor = np.power(distance_to_target / self._MAX_DISTANCE_JOINT_ANGLE, 1/4)
            current_max_tendon_speed = scale_factor * self.robot.MAX_TENDON_VEL
            action = np.multiply(current_max_tendon_speed, action)

        else:
            action = np.multiply(self.robot.MAX_TENDON_VEL, action)

        action = action.tolist()

        new_state = self._ros_proxy.forward_step_command(action)
        self._last_state = new_state
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(current_state=new_state, goal_state=self._goal_state, info=info)
        done = self._did_complete_successfully(current_state=new_state, goal_state=self._goal_state)
        if done:
            print("#############GOAL REACHED#############")

            self._set_new_goal()

        return obs, reward, done, info

    def _make_obs(self, robot_state: MsjRobotState):
        return np.concatenate([
            robot_state.joint_angle,
            robot_state.joint_vel,
            self._goal_state.joint_angle,
        ])

    def reset(self):
        self._ros_proxy.forward_reset_command()
        self._last_state = self._ros_proxy.read_state()
        return self._make_obs(robot_state=self._last_state)

    def render(self, mode='human'):
        pass

    def compute_reward(self, current_state: MsjRobotState, goal_state: MsjRobotState, info=None):

        current_state = self._normalize_state(current_state)
        goal_state = self._normalize_state(goal_state)
        reward = -np.exp(_l2_distance(current_state.joint_angle, goal_state.joint_angle))

        if self._joint_vel_penalty:
            normed_joint_vel = np.linalg.norm(current_state.joint_vel - goal_state.joint_vel)
            reward = (normed_joint_vel+1) * (reward-np.exp(reward))

        if not current_state.is_feasible:
            reward -= np.abs(self._PENALTY_FOR_TOUCHING_BOUNDARY)

        if self._did_complete_successfully(current_state=current_state, goal_state=goal_state) and \
           self._is_agent_getting_bonus_for_reaching_goal:
            reward += self._BONUS_FOR_REACHING_GOAL

        assert self.reward_range[0] <= reward <= self.reward_range[1], \
            "'{}' not between '{}' and '{}'".format(reward, self.reward_range[0], self.reward_range[1])

        return reward

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        new_joint_angle = goal_joint_angle if goal_joint_angle is not None \
            else self._ros_proxy.get_new_goal_joint_angles()
        self._goal_state = MsjRobotState(joint_angle=new_joint_angle,
                                         joint_vel=self._GOAL_JOINT_VEL,
                                         is_feasible=True)

    def _did_complete_successfully(self, current_state: MsjRobotState, goal_state: MsjRobotState) -> bool:
        l2_distance = _l2_distance(current_state.joint_angle, goal_state.joint_angle)
        is_close = bool(l2_distance < self._l2_distance_for_success())  # cast from numpy.bool to bool
        is_moving_slow = all(current_state.joint_vel < self._MAX_TENDON_VEL_FOR_SUCCESS)
        return is_close and is_moving_slow

    def _l2_distance_for_success(self):
        return _l2_distance(-self.robot.JOINT_ANGLE_BOUNDS, self.robot.JOINT_ANGLE_BOUNDS) / 500

    def _normalize_state(self, current_state: MsjRobotState):
        return MsjRobotState(
            joint_angle=current_state.joint_angle / self.robot.JOINT_ANGLE_BOUNDS,
            joint_vel=current_state.joint_vel / self.robot.JOINT_VEL_BOUNDS,
            is_feasible=current_state.is_feasible,
        )