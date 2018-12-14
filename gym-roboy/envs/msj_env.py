import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MSJRobotState:
    def __init__(self, q_position, q_velocity):
        assert len(q_position) == MSJRobot.DIM_Q_POSITION
        assert len(q_velocity) == MSJRobot.DIM_Q_VELOCITY
        self.q_position = q_position
        self.q_velocity = q_velocity


class MSJRobot:
    # ROS topics and stuff
    # position and velocity of the end effector
    # TO-DO subscribe to the q and qdot from ROS1_bridge

    DIM_ACTION = 8
    DIM_Q_POSITION = 3
    DIM_Q_VELOCITY = DIM_Q_POSITION

    @staticmethod
    def read_state() -> MSJRobotState:
        return MSJRobotState(
            q_position=np.random.random(MSJRobot.DIM_Q_POSITION),
            q_velocity=np.random.random(MSJRobot.DIM_Q_VELOCITY),
        )

    @staticmethod
    def forward_action_and_get_new_state(action) -> MSJRobotState:
        assert len(action) == MSJRobot.DIM_ACTION
        return MSJRobot.read_state()


class MsjEnv(gym.GoalEnv):
    #metadata = {'render.modes': ['human']}
    def __init__(self):
        self.seed()
        #self._env_setup(initial_qpos=initial_qpos)
        #self.initial_state = copy.deepcopy(self.sim.get_state())

        #max limit position for q
        self.max_position = 3.14

        #2cm/s for ldot
        self.max_speed = 0.02

        #why do we need a sample goal -> desired goal for the training environment
        #sample goal is the q given
        self.goal_qpos = self._sample_goal()

        #obs is a dictionary which has observation, achieved goal and desired goal. Achieved goal is useful for HER algorithm
        obs = self._make_obs(robot_state=MSJRobot.read_state())

        #ldot is the action space. -2cm/s to 2cm/s
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(MSJRobot.DIM_ACTION,), dtype='float32')
        #subscribe to the q and qdot. MSJ has 3 DOF q is a vector of 3. qdot has the same shape as q.
        #self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        #enter the true joint limits
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
    # A function to initialize the random generator

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # set the topic on of target poses
        # unpause the simulation, implement a ROS2 service to command CARDSflow
        new_state = MSJRobot.forward_action_and_get_new_state(action)
        obs = self._make_obs(robot_state=new_state)

        done = False  # infinite horizon
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal_qpos),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal_qpos, info)
        return obs, reward, done, info

    def _set_action(self, action):
        assert action.shape == (8,)
        #Every ldot should be able to set its own motor.
        #Create controller on KinDyn that directly take ldot as a command
        pass

    def reset(self):
        #resetSim
        #unpauseSim
        #check topic publishers connection
        pass

    def _make_obs(self, robot_state):
        full_obs = np.concatenate(
            [robot_state.q_position, robot_state.q_velocity, self.goal_qpos]
        )
        return {
            'observation': full_obs,
            'achieved_goal': robot_state.q_position.copy(),
            'desired_goal': self.goal_qpos.copy(),
        }

    def render(self, mode='human'):
        pass

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return (achieved_goal-desired_goal) ** 2

    def _sample_goal(self):
        return np.random.random(MSJRobot.DIM_Q_POSITION)

    def _is_success(self, param, goal_qpos):
        pass
