import numpy as np
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset
from .msj_robot_state import MsjRobotState


class MsjROSProxy:
    """
    This interface defines how the MsjEnv interacts with the Msj Robot.
    One implementation will use the ROS1 service over ROS2 bridge.
    """
    def read_state(self) -> MsjRobotState:
        raise NotImplementedError

    def forward_step_command(self, action) -> MsjRobotState:
        raise NotImplementedError

    def forward_reset_command(self) -> MsjRobotState:
        raise NotImplementedError


class MockMsjROSProxy(MsjROSProxy):
    """This implementation is a mock for unit testing purposes."""

    def __init__(self):
        self._state = MsjRobotState.new_random_state()

    def read_state(self) -> MsjRobotState:
        return self._state

    def forward_step_command(self, action) -> MsjRobotState:
        assert len(action) == MsjRobotState.DIM_ACTION
        if np.allclose(action, 0):
            return self._state
        return MsjRobotState.new_random_state()

    def forward_reset_command(self) -> MsjRobotState:
        self._state = MsjRobotState.new_zero_state()
        return self._state


class MsjROSBridgeProxy(MsjROSProxy):

    _RCLPY_INITIALIZED = False

    def __init__(self, timeout_secs: int = 2):
        if not self._RCLPY_INITIALIZED:
            rclpy.init()
            MsjROSBridgeProxy._RCLPY_INITIALIZED = True
        self._timeout_secs = timeout_secs
        self.node = rclpy.create_node('gym_client')
        self.step_client = self.node.create_client(GymStep, 'gym_step')
        self.reset_client = self.node.create_client(GymReset, 'gym_reset')
        self._step_size = 0.1

    def forward_reset_command(self):
        request = GymStep.Request()
        request.step_size = self._step_size
        future = self.reset_client.call_async(request)
        self._wait_until_future_complete_or_timeout(future)
        return self._make_robot_state(service_response=future.result())

    @staticmethod
    def _make_robot_state(service_response) -> MsjRobotState:
        return MsjRobotState(joint_angle=service_response.q[3:],
                             joint_vel=service_response.qdot[3:])

    def forward_step_command(self, action):
        req = GymStep.Request()
        req.set_points = action
        req.step_size = self._step_size
        future = self.step_client.call_async(req)
        self._wait_until_future_complete_or_timeout(future)
        return self._make_robot_state(future.result())

    def _wait_until_future_complete_or_timeout(self, future):
        if not self.step_client.wait_for_service(timeout_sec=self._timeout_secs):
            raise TimeoutError("ROS communication timed out")
        rclpy.spin_until_future_complete(self.node, future)

    def read_state(self):
        req = GymStep.Request()
        future = self.step_client.call_async(req)
        self._wait_until_future_complete_or_timeout(future)
        return self._make_robot_state(future.result())
