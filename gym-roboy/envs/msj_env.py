
# coding: utf-8

# In[8]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding


# In[12]:


class MsjEnv(gym.GoalEnv):
    #metadata = {'render.modes': ['human']}
    def __init__(self, n_actions):
        self.seed()
        #self._env_setup(initial_qpos=initial_qpos)
        #self.initial_state = copy.deepcopy(self.sim.get_state())

        #max limit position for q
        self.max_position = 3.14

        #2cm/s for ldot
        self.max_speed = 0.02

        #why do we need a sample goal -> desired goal for the training environment
        #sample goal is the q given
        self.goal = self._sample_goal()

        #obs is a dictionary which has observation, achieved goal and desired goal. Achieved goal is useful for HER algorithm
        obs = self._get_obs()

        #ldot is the action space. -2cm/s to 2cm/s
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(8,), dtype='float32')
        
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
        self._set_action(action) #set the topic on of target poses
        self.sim.step() #unpause the simulation, implement a ROS2 service to command CARDSflow
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
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
    

    def _get_obs():
        #ROS topics and stuff 
        #position and velocity of the end effector
        #TO-DO subscribe to the q and qdot from ROS1_bridge
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
        
    
    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()
    
    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


