import numpy as np
from gym import spaces
from typeguard import typechecked


class MsjRobotState:
    DIM_ACTION = 8
    MAX_TENDON_VEL = 0.02  # cm/s

    DIM_JOINT_ANGLE = 3
    JOINT_ANGLE_BOUNDS = np.ones(DIM_JOINT_ANGLE) * np.pi
    JOINT_VEL_BOUNDS = np.ones(DIM_JOINT_ANGLE) * np.pi / 6  # 30 deg/sec

    ACTION_SPACE = spaces.Box(low=-1, high=1, shape=(DIM_ACTION,), dtype='float32')

    @typechecked
    def __init__(self, joint_angle, joint_vel, is_feasible: bool):
        assert len(joint_angle) == self.DIM_JOINT_ANGLE
        assert len(joint_vel) == self.DIM_JOINT_ANGLE
        self.joint_angle = np.array(joint_angle)
        self.joint_vel = np.array(joint_vel)
        self.is_feasible = is_feasible

    @classmethod
    def new_random_state(cls):
        return cls(joint_angle=np.random.random(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.random.random(cls.DIM_JOINT_ANGLE),
                   is_feasible=True)

    @classmethod
    def new_zero_state(cls):
        return cls(joint_angle=np.zeros(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.zeros(cls.DIM_JOINT_ANGLE),
                   is_feasible=True)

    @classmethod
    def new_random_zero_vel_state(cls):
        return cls(joint_angle=np.random.random(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.zeros(cls.DIM_JOINT_ANGLE),
                   is_feasible=True)

    @classmethod
    def new_random_zero_angle_state(cls):
        return cls(joint_angle=np.zeros(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.random.random(cls.DIM_JOINT_ANGLE),
                   is_feasible=True)

    @classmethod
    def interpolate(cls, state1, state2):
        assert isinstance(state1, cls) and isinstance(state2, cls)
        return cls(joint_angle=(state1.joint_angle + state2.joint_angle) / 2,
                   joint_vel=(state1.joint_vel + state2.joint_vel) / 2,
                   is_feasible=state1.is_feasible and state2.is_feasible)
