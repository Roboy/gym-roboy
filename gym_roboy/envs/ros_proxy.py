import time
from datetime import datetime
from typing import List

import numpy as np
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset
from roboy_simulation_msgs.srv import GymGoal
from std_msgs.msg import Float32
from typeguard import typechecked

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

    def get_new_goal_joint_angles(self):
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

    def get_new_goal_joint_angles(self):
        return MsjRobotState.new_random_state().joint_angle


class MsjROSBridgeProxy(MsjROSProxy):

    _RCLPY_INITIALIZED = False

    def __init__(self, process_idx: int = 1, timeout_secs: int = 20):
        if not self._RCLPY_INITIALIZED:
            rclpy.init()
            MsjROSBridgeProxy._RCLPY_INITIALIZED = True
        self._timeout_secs = timeout_secs
        self._step_size = 0.1
        self.node = rclpy.create_node('gym_rosnode')
        self._create_ros_client(process_idx)
        self._last_time_gym_goal_service_was_called = datetime.now()

    def _create_ros_client(self, process_idx: int):
        self.step_client = self.node.create_client(GymStep, '/instance' + str(process_idx) + '/gym_step')
        self.reset_client = self.node.create_client(GymReset, '/instance' + str(process_idx) + '/gym_reset')
        self.goal_client = self.node.create_client(GymGoal, '/instance' + str(process_idx) + '/gym_goal')

    def _log_robot_state(self, robot_state):
        q_pos = robot_state.q
        q_vel = robot_state.qdot
        qpos_str = str(q_pos).strip('[]')
        qvel_str = str(q_vel).strip('[]')
        self.node.get_logger().info("joint angles: %s" % qpos_str)
        self.node.get_logger().info("joint velocity: %s" % qvel_str)

    def forward_reset_command(self):
        self._check_service_available_or_timeout(self.reset_client)
        request = GymStep.Request()
        request.step_size = self._step_size
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        return self._make_robot_state(service_response=future.result())

    @staticmethod
    def _make_robot_state(service_response) -> MsjRobotState:
        feasible = service_response.feasible if hasattr(service_response, "feasible") else True
        return MsjRobotState(
            joint_angle=service_response.q,
            joint_vel=service_response.qdot,
            is_feasible=feasible,
        )

    @typechecked
    def forward_step_command(self, action: List[float]):
        self._check_service_available_or_timeout(self.step_client)

        request = GymStep.Request()
        request.set_points = action
        request.step_size = self._step_size

        future = self.step_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        res = future.result()

        return self._make_robot_state(res)

    def _check_service_available_or_timeout(self, client) -> None:
        if not client.wait_for_service(timeout_sec=self._timeout_secs):
            raise TimeoutError("ROS communication timed out")

    def read_state(self):
        self._check_service_available_or_timeout(self.step_client)
        req = GymStep.Request()
        future = self.step_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return self._make_robot_state(future.result())

    def get_new_goal_joint_angles(self):
        #self.node.get_logger().info("Reached goal joint angles: " + str(self.read_state().joint_angle))
        self._delay_if_necessary()
        self._check_service_available_or_timeout(self.goal_client)
        req = GymGoal.Request()
        future = self.goal_client.call_async(req)
        self._last_time_gym_goal_service_was_called = datetime.now()
        rclpy.spin_until_future_complete(self.node, future)
        res = future.result()
        if res is not None:
            self.node.get_logger().info("feasible: " + str(res.q))
        return res.q

    def _delay_if_necessary(self) -> None:
        """If the /gym_goal service gets called within a second it
        delivers the same joint angle. Delaying is a quick fix."""
        now = datetime.now()
        seconds_since_last_service_call = (now - self._last_time_gym_goal_service_was_called).total_seconds()
        min_seconds_between_calls = 1.1
        if seconds_since_last_service_call <= min_seconds_between_calls:
            time.sleep(min_seconds_between_calls - seconds_since_last_service_call)
