import numpy as np
from gym import spaces
from . import RoboyRobot


class MsjRobot(RoboyRobot):

    _DIM_JOINT_ANGLE = 3
    _JOINT_ANGLE_SPACE = spaces.Box(low=-np.pi, high=np.pi, shape=(_DIM_JOINT_ANGLE,), dtype="float32")
    _JOINT_VEL_SPACE = spaces.Box(low=-np.pi/6, high=np.pi/6, shape=(_DIM_JOINT_ANGLE,), dtype="float32")

    _DIM_ACTION = 8
    _MAX_TENDON_VEL = 0.02  # cm/s

    _MAX_TENDON_LENGHT = 0.3  # cm
    _ACTION_SPACE = spaces.Box(low=-_MAX_TENDON_LENGHT, high=_MAX_TENDON_LENGHT, shape=(_DIM_ACTION,), dtype='float32')

    @classmethod
    def get_action_space(cls) -> spaces.Box:
        return cls._ACTION_SPACE

    @classmethod
    def get_joint_angles_space(cls) -> spaces.Box:
        return cls._JOINT_ANGLE_SPACE

    @classmethod
    def get_joint_vels_space(cls) -> spaces.Box:
        return cls._JOINT_VEL_SPACE
