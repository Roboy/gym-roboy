from typing import Tuple
import numpy as np
import gym
from gym import spaces
from .simulations import SimulationClient
from typeguard import typechecked
from .robots import RobotState, RoboyRobot


class RoboyEnv(gym.GoalEnv):

    def __init__(self, simulation_client: SimulationClient, seed: int = None,
                 joint_vel_penalty: bool = False,
                 is_agent_getting_bonus_for_reaching_goal: bool = True):
        self.seed(seed)
        self._simulation_client = simulation_client
        self._joint_vel_penalty = joint_vel_penalty
        self._is_agent_getting_bonus_for_reaching_goal = is_agent_getting_bonus_for_reaching_goal
        self._robot = robot = simulation_client.robot
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
            dtype="float32",
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

        action = _rescale_from_one_space_to_other(
            input_space=self.action_space,
            output_space=self._robot.get_action_space(),
            input_val=np.array(action)).tolist()

        new_state = self._simulation_client.forward_step_command(action)
        self.step_num += 1
        self._last_state = new_state
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(current_state=new_state, goal_state=self._goal_state, info=info)
        done = self._did_reach_goal(current_state=new_state, goal_state=self._goal_state) or \
               self._reached_max_steps()
        if done:
            self._set_new_goal()

        return obs, reward, done, info

    def _reached_max_steps(self) -> bool:
        return self.step_num > self._MAX_EPISODE_LENGTH

    def _make_obs(self, robot_state: RobotState):
        return np.concatenate([
            robot_state.joint_angles,
            robot_state.joint_vels,
            self._goal_state.joint_angles,
        ])

    def reset(self):
        self._simulation_client.forward_reset_command()
        self._last_state = self._simulation_client.read_state()
        self.step_num = 1
        self._set_new_goal()
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

        if self._did_reach_goal(current_state=current_state, goal_state=goal_state) and \
           self._is_agent_getting_bonus_for_reaching_goal:
            reward += self._BONUS_FOR_REACHING_GOAL

        assert self.reward_range[0] <= reward <= self.reward_range[1], \
            "'{}' not between '{}' and '{}'".format(reward, self.reward_range[0], self.reward_range[1])

        return float(reward)  # cast from numpy.float to float

    def seed(self, seed=None):
        np.random.seed(seed)

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        new_joint_angle = goal_joint_angle if goal_joint_angle is not None \
            else self._simulation_client.get_new_goal_joint_angles()
        self._goal_state = self._robot.new_state(joint_angle=new_joint_angle,
                                                 joint_vel=self._GOAL_JOINT_VEL,
                                                 is_feasible=True)

    def _did_reach_goal(self, current_state: RobotState, goal_state: RobotState) -> bool:
        angles_l2_distance = _l2_distance(current_state.joint_angles, goal_state.joint_angles)
        angles_are_close = bool(angles_l2_distance < self._MAX_DISTANCE_JOINT_ANGLE/500)  # cast from numpy.bool to bool

        vels_l2_distance = _l2_distance(current_state.joint_vels, goal_state.joint_vels)
        vels_are_close = bool(vels_l2_distance < self._MAX_DISTANCE_JOINT_VELS/100)  # fraction not yet tuned

        if angles_are_close and vels_are_close:
            print("#############GOAL REACHED#############")
        return angles_are_close and vels_are_close


def _l2_distance(joint_angle1, joint_angle2):
    subtract = np.subtract(joint_angle1, joint_angle2)
    subtract[np.isnan(subtract)] = 0  # np.inf - np.inf returns np.nan, but should be 0
    return np.linalg.norm(subtract, ord=2)


@typechecked
def _rescale_from_one_space_to_other(
        input_val: np.ndarray, input_space: spaces.Box, output_space: spaces.Box) -> np.ndarray:
    """
    Transforms a point in a input space to a corresponding point in the
    output space. The output point is as far away from the output
    boundaries as the input point is from the input boundaries.
    :param input_space: bounded Box
    :param output_space: bounded Box
    :param input_val: The input point
    :return: The output point in the output space
    """
    assert input_space.shape == output_space.shape
    assert input_space.contains(input_val)
    slope = (output_space.high-output_space.low) / (input_space.high-input_space.low)
    return slope * (input_val - input_space.high) + output_space.high
