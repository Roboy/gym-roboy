import numpy as np

import gym
from gym import spaces
from . import MsjROSProxy, MsjROSBridgeProxy, MsjRobotState


def _l2_distance(joint_angle1, joint_angle2):
    return np.linalg.norm(np.subtract(joint_angle1, joint_angle2), ord=2)


class MsjEnv(gym.GoalEnv):

    _max_joint_angle = np.pi
    _max_tendon_speed = 0.02  # cm/s
    # TODO: joint velocities are not constrained to [-pi, pi], only angles.
    # 3 * DIM_JOINT_ANGLE from the observation = DIM_current_joint_velocity + DIM_current_joints + DIM_goal_joints
    observation_space = spaces.Box(-_max_joint_angle, _max_joint_angle,
                                   shape=(3*MsjRobotState.DIM_JOINT_ANGLE,), dtype='float32')
    action_space = spaces.Box(
        low=-1,
        high=1,
        shape=(MsjRobotState.DIM_ACTION,)
        , dtype='float32'
    )

    #print("The action space is: " + str(action_space))

    reward_range = (-_l2_distance(observation_space.low, observation_space.high),
                    -_l2_distance(observation_space.low, observation_space.low))

    def __init__(self, ros_proxy: MsjROSProxy=MsjROSBridgeProxy(), seed: int = None):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._set_new_goal()

    def step(self, action):
        action = self._max_tendon_speed * np.clip(action, self.action_space.low, self.action_space.high)
        action = action.tolist()
        #print(action)
        new_state = self._ros_proxy.forward_step_command(action)
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(new_state.joint_angle, self._goal_joint_angle, info)
        #print("step reward:", reward)
        done = self._did_reach_goal(actual_joint_angle=new_state.joint_angle)
        if done:
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

    def compute_reward(self, achieved_goal, desired_goal, info):
        current_joint_angle = achieved_goal
        reward = -_l2_distance(current_joint_angle, desired_goal)
        debug_msg = str(reward) + " not in: " + str(self.reward_range) + \
                    " current goal: " + str(achieved_goal) + \
                    " desired goal: " + str(desired_goal)
        # TODO: This assert should work
        # assert self.reward_range[0] < reward < self.reward_range[1], debug_msg
        return reward

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        if goal_joint_angle is not None:
            self._goal_joint_angle = goal_joint_angle
            return
        new_joint_angle = self._ros_proxy.get_new_goal_joint_angles()
        self._goal_joint_angle = new_joint_angle
        
    def _did_reach_goal(self, actual_joint_angle) -> bool:
        l2_distance = _l2_distance(actual_joint_angle, self._goal_joint_angle)
        return bool(l2_distance < self._l2_distance_for_success) # bool for comparison to a numpy bool

    @property
    def _l2_distance_for_success(self):
        return _l2_distance(self.observation_space.low, self.observation_space.high) / 1000  # 100 seems reasonable
