from gym import spaces
from typeguard import typechecked
import numpy as np


class RobotState:
    @typechecked
    def __init__(self, joint_angles, joint_vels, is_feasible: bool):
        self.joint_angles = np.array(joint_angles)
        self.joint_vels = np.array(joint_vels)
        self.is_feasible = is_feasible

    @classmethod
    def interpolate(cls, state1, state2):
        assert isinstance(state1, cls) and isinstance(state2, cls)
        return cls(joint_angles=(state1.joint_angles + state2.joint_angles) / 2,
                   joint_vels=(state1.joint_vels + state2.joint_vels) / 2,
                   is_feasible=state1.is_feasible and state2.is_feasible)


class RoboyRobot:

    @classmethod
    def get_action_space(cls) -> spaces.Box:
        raise NotImplementedError

    @classmethod
    def get_joint_angles_space(cls) -> spaces.Box:
        raise NotImplementedError

    @classmethod
    def get_joint_vels_space(cls) -> spaces.Box:
        raise NotImplementedError

    @classmethod
    def new_random_state(cls) -> RobotState:
        return RobotState(joint_angles=cls.get_joint_angles_space().sample(),
                          joint_vels=cls.get_joint_angles_space().sample(),
                          is_feasible=True)

    @classmethod
    def new_zero_state(cls) -> RobotState:
        return RobotState(joint_angles=np.zeros(cls.get_joint_angles_space().shape),
                          joint_vels=np.zeros(cls.get_joint_vels_space().shape),
                          is_feasible=True)

    @classmethod
    def new_random_zero_vels_state(cls) -> RobotState:
        return RobotState(joint_angles=cls.get_joint_angles_space().sample(),
                          joint_vels=np.zeros(cls.get_joint_vels_space().shape),
                          is_feasible=True)

    @classmethod
    def new_max_state(cls) -> RobotState:
        return RobotState(joint_angles=cls.get_joint_angles_space().high,
                          joint_vels=cls.get_joint_vels_space().high,
                          is_feasible=False)

    @classmethod
    def new_min_state(cls) -> RobotState:
        return RobotState(joint_angles=cls.get_joint_angles_space().low,
                          joint_vels=cls.get_joint_vels_space().low,
                          is_feasible=False)

    @classmethod
    def new_random_zero_angles_state(cls) -> RobotState:
        return RobotState(joint_angles=np.zeros(cls.get_joint_angles_space().shape),
                          joint_vels=cls.get_joint_vels_space().sample(),
                          is_feasible=True)

    @classmethod
    @typechecked
    def new_state(cls, joint_angle, joint_vel, is_feasible: bool) -> RobotState:
        joint_angle = np.array(joint_angle) if not isinstance(joint_angle, np.ndarray) else joint_angle
        joint_vel = np.array(joint_vel) if not isinstance(joint_vel, np.ndarray) else joint_vel
        assert cls.get_joint_angles_space().contains(joint_angle)
        # assert cls.get_joint_vels_space().contains(joint_vels) TODO: This assert should work
        return RobotState(joint_angles=joint_angle, joint_vels=joint_vel, is_feasible=is_feasible)

    def normalize_state(self, state: RobotState) -> RobotState:
        max_angles = self.get_joint_angles_space().high
        min_angles = self.get_joint_angles_space().low

        max_vels = self.get_joint_vels_space().high
        min_vels = self.get_joint_vels_space().low

        return RobotState(
            joint_angles=self._normalize_between_minus1_and1(state.joint_angles, max_angles, min_angles),
            joint_vels=self._normalize_between_minus1_and1(state.joint_vels, max_vels, min_vels),
            is_feasible=state.is_feasible,
        )

    @staticmethod
    def _normalize_between_minus1_and1(val, max_val, min_val):
        return (2*val - max_val - min_val) / (max_val - min_val)
