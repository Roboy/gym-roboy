import numpy as np

from ..robots import RobotState, RoboyRobot


class SimulationClient:
    """
    This interface defines how the RoboyEnv interacts with the Roboy Robot.
    One implementation will use the ROS1 service over ROS2 bridge.
    """
    robot = RoboyRobot()

    def read_state(self) -> RobotState:
        raise NotImplementedError

    def forward_step_command(self, action) -> RobotState:
        raise NotImplementedError

    def forward_reset_command(self) -> RobotState:
        raise NotImplementedError

    def get_new_goal_joint_angles(self) -> np.ndarray:
        raise NotImplementedError


class StubSimulationClient(SimulationClient):
    """This implementation is a stub for unit testing purposes."""

    def __init__(self, robot: RoboyRobot):
        self.robot = robot
        self._state = self.robot.new_random_state()

    def read_state(self) -> RobotState:
        return self._state

    def forward_step_command(self, action) -> RobotState:
        assert self.robot.get_action_space().shape[0] == len(action)
        if np.allclose(action, 0):
            return self._state
        return self.robot.new_random_state()

    def forward_reset_command(self) -> RobotState:
        self._state = self.robot.new_zero_state()
        return self._state

    def get_new_goal_joint_angles(self):
        return self.robot.new_random_state().joint_angles
