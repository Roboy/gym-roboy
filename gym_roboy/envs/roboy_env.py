from typing import Tuple
import numpy as np
import gym
from gym import spaces
from . import ROSProxy
from .robots import RobotState, RoboyRobot


def _l2_distance(joint_angle1, joint_angle2):
    subtract = np.subtract(joint_angle1, joint_angle2)
    subtract[np.isnan(subtract)] = 0  # np.inf - np.inf returns np.nan, but should be 0
    return np.linalg.norm(subtract, ord=2)


class RoboyEnv(gym.GoalEnv):

    def __init__(self, ros_proxy: ROSProxy, seed: int = None,
                 joint_vel_penalty: bool = False,
                 is_tendon_vel_dependent_on_distance: bool = True,
                 is_agent_getting_bonus_for_reaching_goal: bool = True):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._joint_vel_penalty = joint_vel_penalty
        self._is_tendon_vel_dependent_on_distance = is_tendon_vel_dependent_on_distance
        self._is_agent_getting_bonus_for_reaching_goal = is_agent_getting_bonus_for_reaching_goal
        self._robot = robot = ros_proxy.robot
        self._last_state = None  # type: RobotState
        self._goal_state = None  # type: RobotState

        self._GOAL_JOINT_VEL = robot.new_zero_state().joint_angles
        self._MAX_DISTANCE_JOINT_ANGLE = _l2_distance(robot.get_joint_angles_space().low, robot.get_joint_angles_space().high)
        self._MAX_DISTANCE_JOINT_VELS = _l2_distance(robot.get_joint_vels_space().low, robot.get_joint_vels_space().high)
        self._PENALTY_FOR_TOUCHING_BOUNDARY = 1
        self._BONUS_FOR_REACHING_GOAL = 1000
        self._MAX_EPISODE_LENGTH = 1000

        self.reward_range = self._create_reward_range(robot=robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=robot.get_action_space().shape, dtype="float32")
        self.observation_space = spaces.Box(
            low=np.concatenate((robot.get_joint_angles_space().low, robot.get_joint_vels_space().low, robot.get_joint_angles_space().low)),
            high=np.concatenate((robot.get_joint_angles_space().high, robot.get_joint_vels_space().high, robot.get_joint_angles_space().high)),
        )
        self._set_new_goal()
        self.step_num = 1

    def _create_reward_range(self, robot: RoboyRobot) -> Tuple[float, float]:
        some_state = robot.new_state(
            joint_angle=robot.get_joint_angles_space().high,
            joint_vel=robot.get_joint_vels_space().high, is_feasible=True)
        corresponding_worst_state = robot.new_state(
            joint_angle=robot.get_joint_angles_space().low,
            joint_vel=robot.get_joint_vels_space().low, is_feasible=False)
        max_reward = self.compute_reward(current_state=some_state, goal_state=some_state)
        min_reward = self.compute_reward(current_state=corresponding_worst_state, goal_state=some_state)
        return min_reward, max_reward

    def step(self, action):
        assert self.action_space.contains(action)

        if self._is_tendon_vel_dependent_on_distance:
            distance_to_target = _l2_distance(self._last_state.joint_angles, self._goal_state.joint_angles)
            scale_factor = np.power(distance_to_target / self._MAX_DISTANCE_JOINT_ANGLE, 1/4)
            current_max_tendon_speed = scale_factor * self._robot.get_action_space().high
            action = np.multiply(current_max_tendon_speed, action)

        else:
            action = np.multiply(self._robot.get_action_space().high, action)

        action = action.tolist()

        new_state = self._ros_proxy.forward_step_command(action)
        self.step_num += 1
        self._last_state = new_state
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(current_state=new_state, goal_state=self._goal_state, info=info)
        done = self._did_complete_successfully(current_state=new_state, goal_state=self._goal_state) or self.step_num > self._MAX_EPISODE_LENGTH
        if done:
            self._set_new_goal()

        return obs, reward, done, info

    def _make_obs(self, robot_state: RobotState):
        return np.concatenate([
            robot_state.joint_angles,
            robot_state.joint_vels,
            self._goal_state.joint_angles,
        ])

    def reset(self):
        self._ros_proxy.forward_reset_command()
        self._last_state = self._ros_proxy.read_state()
        self.step_num = 1
        return self._make_obs(robot_state=self._last_state)

    def render(self, mode='human'):
        pass

    def compute_reward(self, current_state: RobotState, goal_state: RobotState, info=None):

        current_state = self._robot.normalize_state(current_state)
        goal_state = self._robot.normalize_state(goal_state)
        reward = -np.exp(_l2_distance(current_state.joint_angles, goal_state.joint_angles))

        if self._joint_vel_penalty:
            normed_joint_vel = np.linalg.norm(current_state.joint_vels - goal_state.joint_vels)
            reward = (normed_joint_vel+1) * (reward-np.exp(reward))

        if not current_state.is_feasible:
            reward -= np.abs(self._PENALTY_FOR_TOUCHING_BOUNDARY)

        if self._did_complete_successfully(current_state=current_state, goal_state=goal_state) and \
           self._is_agent_getting_bonus_for_reaching_goal:
            reward += self._BONUS_FOR_REACHING_GOAL

        assert self.reward_range[0] <= reward <= self.reward_range[1], \
            "'{}' not between '{}' and '{}'".format(reward, self.reward_range[0], self.reward_range[1])

        return float(reward)  # cast from numpy.float to float

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        new_joint_angle = goal_joint_angle if goal_joint_angle is not None \
            else self._ros_proxy.get_new_goal_joint_angles()
        self._goal_state = self._robot.new_state(joint_angle=new_joint_angle,
                                                 joint_vel=self._GOAL_JOINT_VEL,
                                                 is_feasible=True)

    def _did_complete_successfully(self, current_state: RobotState, goal_state: RobotState) -> bool:
        angles_l2_distance = _l2_distance(current_state.joint_angles, goal_state.joint_angles)
        angles_are_close = bool(angles_l2_distance < self._MAX_DISTANCE_JOINT_ANGLE/500)  # cast from numpy.bool to bool

        vels_l2_distance = _l2_distance(current_state.joint_vels, goal_state.joint_vels)
        vels_are_close = bool(vels_l2_distance < self._MAX_DISTANCE_JOINT_VELS/100)  # fraction not yet tuned

        if angles_are_close and vels_are_close:
            print("#############GOAL REACHED#############")
        return angles_are_close and vels_are_close
