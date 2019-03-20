from typing import List
from typeguard import typechecked
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset
from roboy_simulation_msgs.srv import GymGoal

from . import SimulationClient
from ..robots import RobotState, RoboyRobot


class RosSimulationClient(SimulationClient):

    _RCLPY_INITIALIZED = False

    def __init__(self, robot: RoboyRobot, process_idx: int = 1, timeout_secs: int = 2):
        if not self._RCLPY_INITIALIZED:
            rclpy.init()
            RosSimulationClient._RCLPY_INITIALIZED = True
        self.robot = robot
        self._timeout_secs = timeout_secs
        self._step_size = 0.1
        self.node = rclpy.create_node('gym_python_client_node_' + str(process_idx))
        self._create_ros_client(process_idx)

    def _create_ros_client(self, process_idx: int):
        self.step_client = self.node.create_client(GymStep, '/instance' + str(process_idx) + '/gym_step')
        self.reset_client = self.node.create_client(GymReset, '/instance' + str(process_idx) + '/gym_reset')
        self.goal_client = self.node.create_client(GymGoal, '/instance' + str(process_idx) + '/gym_goal')
        self.read_state_client = self.node.create_client(GymStep, '/instance' + str(process_idx) + '/gym_read_state')

    def forward_reset_command(self):
        self._check_service_available_or_timeout(self.reset_client)
        request = GymReset.Request()  #TODO: new service message type for reset service
        request.step_size = self._step_size
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        return self._make_robot_state(service_response=future.result())

    def _make_robot_state(self, service_response) -> RobotState:
        feasible = service_response.feasible if hasattr(service_response, "feasible") else True
        return self.robot.new_state(
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
            raise TimeoutError("ROS communication timed out:", client.srv_name)

    def read_state(self):
        self._check_service_available_or_timeout(self.read_state_client)
        req = GymStep.Request() #TODO: new service type for read_state_client
        future = self.read_state_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return self._make_robot_state(future.result())

    def get_new_goal_joint_angles(self):
        self._check_service_available_or_timeout(self.goal_client)
        req = GymGoal.Request()
        future = self.goal_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        res = future.result()
        if res is not None:
            self.node.get_logger().info("feasible: " + str(res.q))
        return res.q