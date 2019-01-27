import numpy as np


class MsjRobotState:

    DIM_ACTION = 8
    DIM_JOINT_ANGLE = 3

    def __init__(self, joint_angle, joint_vel):
        assert len(joint_angle) == self.DIM_JOINT_ANGLE
        assert len(joint_vel) == self.DIM_JOINT_ANGLE
        self.joint_angle = np.array(joint_angle)
        self.joint_vel = np.array(joint_vel)

    @classmethod
    def new_random_state(cls):
        return cls(joint_angle=np.random.random(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.random.random(cls.DIM_JOINT_ANGLE))

    @classmethod
    def new_zero_state(cls):
        return cls(joint_angle=np.zeros(cls.DIM_JOINT_ANGLE),
                   joint_vel=np.zeros(cls.DIM_JOINT_ANGLE))
