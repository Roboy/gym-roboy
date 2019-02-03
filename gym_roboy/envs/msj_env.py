import numpy as np

import gym
from gym import spaces
from . import MsjROSProxy, MsjROSBridgeProxy, MsjRobotState


def _l2_distance(joint_angle1, joint_angle2):
    return np.linalg.norm(np.subtract(joint_angle1, joint_angle2), ord=2)


class MsjEnv(gym.GoalEnv):

    _max_tendon_speed = 0.02  # cm/s
    _JOINT_ANGLE_BOUNDS = np.ones(MsjRobotState.DIM_JOINT_ANGLE) * np.pi
    _JOINT_VEL_BOUNDS = np.ones(MsjRobotState.DIM_JOINT_ANGLE) * np.inf
    observation_space = spaces.Box(
        low=-np.concatenate((_JOINT_ANGLE_BOUNDS, _JOINT_VEL_BOUNDS, _JOINT_ANGLE_BOUNDS)),
        high=np.concatenate((_JOINT_ANGLE_BOUNDS, _JOINT_VEL_BOUNDS, _JOINT_ANGLE_BOUNDS)),
    )
    action_space = spaces.Box(
        low=-1,
        high=1,
        shape=(MsjRobotState.DIM_ACTION,)
        , dtype='float32'
    )

    def __init__(self, ros_proxy: MsjROSProxy=MsjROSBridgeProxy(), seed: int = None):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._set_new_goal()
        self.reward_range = (
            self.compute_reward(-self._JOINT_ANGLE_BOUNDS, self._JOINT_ANGLE_BOUNDS, info={}),
            self.compute_reward(self._JOINT_ANGLE_BOUNDS, self._JOINT_ANGLE_BOUNDS, info={})
        )

    def step(self, action):
        action = self._max_tendon_speed * np.clip(action, self.action_space.low, self.action_space.high)
        action = action.tolist()
        new_state = self._ros_proxy.forward_step_command(action)
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(new_state.joint_angle, self._goal_joint_angle, info)
        done = self._did_reach_goal(actual_joint_angle=new_state.joint_angle)
        if done:
            print("#############GOAL REACHED#############")
            self._set_new_goal()

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

    @classmethod
    def compute_reward(cls, achieved_goal, desired_goal, info):
        assert len(achieved_goal) == MsjRobotState.DIM_JOINT_ANGLE, str(len(achieved_goal))
        assert len(desired_goal) == MsjRobotState.DIM_JOINT_ANGLE, str(len(desired_goal))
        current_joint_angle = achieved_goal
        reward = -_l2_distance(current_joint_angle, desired_goal)
        assert cls.reward_range[0] <= reward <= cls.reward_range[1]
        return reward

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        if goal_joint_angle is not None:
            self._goal_joint_angle = goal_joint_angle
            return
        new_joint_angle = self._ros_proxy.get_new_goal_joint_angles()
        assert len(new_joint_angle) == MsjRobotState.DIM_JOINT_ANGLE, str(len(new_joint_angle))
        self._goal_joint_angle = new_joint_angle
        
    def _did_reach_goal(self, actual_joint_angle) -> bool:
        l2_distance = _l2_distance(actual_joint_angle, self._goal_joint_angle)
        return bool(l2_distance < self._l2_distance_for_success) # bool for comparison to a numpy bool

    @property
    def _l2_distance_for_success(self):
        return _l2_distance(-self._JOINT_ANGLE_BOUNDS, self._JOINT_ANGLE_BOUNDS) / 1000
