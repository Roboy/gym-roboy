import numpy as np

import gym
from gym import spaces
from . import MsjROSProxy, MsjROSBridgeProxy, MsjRobotState


def _l2_distance(joint_angle1, joint_angle2):
    return np.linalg.norm(np.subtract(joint_angle1, joint_angle2), ord=2)


class MsjEnv(gym.GoalEnv):
    _max_tendon_speed = 0.02  # cm/s

    _max_joint_angle = np.pi
    _max_joint_vel = np.pi / 6  # 30 deg/sec

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
    _GOAL_JOINT_VEL = np.zeros(MsjRobotState.DIM_JOINT_ANGLE)

    def __init__(self, ros_proxy: MsjROSProxy = MsjROSBridgeProxy(), seed: int = None):
        self.seed(seed)
        self._ros_proxy = ros_proxy
        self._set_new_goal()
        worst_state = MsjRobotState(joint_angle=self._JOINT_ANGLE_BOUNDS, joint_vel=self._JOINT_VEL_BOUNDS)
        self.reward_range = (
            self.compute_reward(current_state=worst_state, goal_state=self._goal_state),  # min-reward
            self.compute_reward(current_state=self._goal_state, goal_state=self._goal_state)  # max-reward
        )

    def step(self, action):
        action = self._max_tendon_speed * np.clip(action, self.action_space.low, self.action_space.high)
        action = action.tolist()
        new_state = self._ros_proxy.forward_step_command(action)
        obs = self._make_obs(robot_state=new_state)
        info = {}
        reward = self.compute_reward(current_state=new_state, goal_state=self._goal_state, info=info)
        done = self._did_reach_goal(current_state=new_state)
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

    def compute_reward(self, current_state: MsjRobotState, goal_state: MsjRobotState, info=None):

        normal_current_state = self._normalization(current_state)
        normal_goal_state = self._normalization(goal_state)

        # print(current_state.joint_vel)

        joint_angle_reward = _l2_distance(normal_current_state.joint_angle, normal_goal_state.joint_angle)
        joint_angle_reward = -np.power(joint_angle_reward, 2)
        joint_vel_reward = _l2_distance(normal_current_state.joint_vel, normal_goal_state.joint_vel)
        joint_vel_reward = -np.power(joint_vel_reward, 2)
        total_reward = joint_angle_reward + joint_vel_reward
        # print(total_reward)
        # assert cls.reward_range[0] <= total_reward <= cls.reward_range[1]
        return total_reward

    def _set_new_goal(self, goal_joint_angle=None):
        """If the input goal is None, we choose a random one."""
        self._goal_joint_angle = goal_joint_angle if goal_joint_angle is not None \
            else self._ros_proxy.get_new_goal_joint_angles()
        self._goal_state = MsjRobotState(joint_angle=self._goal_joint_angle,
                                         joint_vel=self._GOAL_JOINT_VEL)

    def _did_reach_goal(self, current_state: MsjRobotState) -> bool:
        joint_l2_distance = _l2_distance(current_state.joint_angle, self._goal_state.joint_angle)
        bool_joint = joint_l2_distance < self._l2_distance_joint_for_success()
        bool_vel = np.all(np.abs(current_state.joint_vel) < np.array([0.005]*3))

        if bool_joint and bool_vel:
            done = True

        else:
            done = False

        return done  # bool for comparison to a numpy bool

    def _l2_distance_joint_for_success(self):
        return _l2_distance(-self._JOINT_ANGLE_BOUNDS, self._JOINT_ANGLE_BOUNDS) / 1000

    def _normalization(self, robot_state: MsjRobotState):
        normal_current_state = MsjRobotState(joint_angle=[0]*MsjRobotState.DIM_JOINT_ANGLE, joint_vel=[0]*MsjRobotState.DIM_JOINT_ANGLE)
        normal_current_state.joint_angle = robot_state.joint_angle / self._max_joint_angle
        normal_current_state.joint_vel = robot_state.joint_vel / self._max_joint_vel

        return normal_current_state
