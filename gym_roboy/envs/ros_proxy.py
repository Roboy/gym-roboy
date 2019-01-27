import numpy as np
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset
from roboy_simulation_msgs.srv import GymGoal
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32



class MsjRobotState:
    def __init__(self, joint_angle, joint_vel):
        assert len(joint_angle) == MsjROSProxy.DIM_JOINT_ANGLE
        assert len(joint_vel) == MsjROSProxy.DIM_JOINT_ANGLE
        self.joint_angle = np.array(joint_angle)
        self.joint_vel = np.array(joint_vel)


class MsjROSProxy:
    """
    This interface defines how the MsjEnv interacts with the Msj Robot.
    One implementation will use the ROS1 service over ROS2 bridge.
    """

    DIM_ACTION = 8
    DIM_JOINT_ANGLE = 3

    def read_state(self) -> MsjRobotState:
        raise NotImplementedError

    def forward_step_command(self, action) -> MsjRobotState:
        raise NotImplementedError

    def forward_reset_command(self) -> MsjRobotState:
        raise NotImplementedError

    def forward_new_goal(self, _goal_joint_angle):
        raise NotImplementedError


class MockMsjROSProxy(MsjROSProxy):
    """This implementation is a mock for unit testing purposes."""

    def __init__(self):
        self._state = MsjRobotState(
            joint_angle=np.random.random(self.DIM_JOINT_ANGLE),
            joint_vel=np.random.random(self.DIM_JOINT_ANGLE),
        )

    def read_state(self) -> MsjRobotState:
        return self._state

    def forward_step_command(self, action) -> MsjRobotState:
        assert len(action) == self.DIM_ACTION
        return self._state

    def forward_reset_command(self) -> MsjRobotState:
        self._state = MsjRobotState(
            joint_angle=np.zeros(self.DIM_JOINT_ANGLE),
            joint_vel=np.zeros(self.DIM_JOINT_ANGLE),
        )
        return self._state


class MsjROSBridgeProxy(MsjROSProxy):
    _step_size = 0.01

    def __init__(self):
        try:
            rclpy.init()
        except:
            pass
        self.node = rclpy.create_node('gym_client')
        self.step_cli = self.node.create_client(GymStep, 'gym_step')
        self.reset_cli = self.node.create_client(GymReset, 'gym_reset')
        self.goal_cli = self.node.create_client(GymGoal, 'gym_goal')
        self.sphere_axis0 = self.node.create_publisher(msg_type=Float32, topic="/sphere_axis0/sphere_axis0/target")
        self.sphere_axis1 = self.node.create_publisher(msg_type=Float32, topic="/sphere_axis1/sphere_axis1/target")
        self.sphere_axis2 = self.node.create_publisher(msg_type=Float32, topic="/sphere_axis2/sphere_axis2/target")

    def _log_robot_state(self, robot_state):
        q_pos = robot_state.q
        q_vel = robot_state.qdot
        qpos_str = str(q_pos).strip('[]')
        qvel_str = str(q_vel).strip('[]')
        self.node.get_logger().info("joint angles: %s" % qpos_str)
        self.node.get_logger().info("joint velocity: %s" % qvel_str)

    def _check_service(self, srv):
        while not srv.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')
        return True

    def forward_reset_command(self):
        while self._check_service(self.reset_cli):
            req = GymStep.Request()
            future = self.reset_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node,future)
            res = future.result()
            if future.result() is not None:
                self._log_robot_state(res)
            return self._make_robot_state(service_response=res)

    @staticmethod
    def _make_robot_state(service_response) -> MsjRobotState:
        return MsjRobotState(joint_angle=service_response.q,
                             joint_vel=service_response.qdot)

    def forward_step_command(self, action):
        while self._check_service(self.step_cli):
            req = GymStep.Request()
            req.set_points = action
            req.step_size = self._step_size
            future = self.step_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node,future)
            res = future.result()
            if res is not None:
                self._log_robot_state(res)
                if not res.feasible:
                    return self.forward_reset_command()
            return self._make_robot_state(res)

    def read_state(self):
        while self._check_service(self.step_cli):
            req = GymStep.Request()
            future = self.step_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            res = future.result()
            return self._make_robot_state(res)

    def forward_new_goal(self, goal_joint_angle):
        assert len(goal_joint_angle) == 3

        msg0 = Float32()
        msg1 = Float32()
        msg2 = Float32()
        msg0.data = goal_joint_angle[0]
        msg1.data = goal_joint_angle[1]
        msg2.data = goal_joint_angle[2]

        self.sphere_axis0.publish(msg0)
        self.sphere_axis1.publish(msg1)
        self.sphere_axis2.publish(msg2)

    def set_new_goal(self):
        while self._check_service(self.goal_cli):
            req = GymGoal.Request()
            future = self.goal_cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)
            res = future.result()
            if res is not None:
                self.node.get_logger().info("feasible: " + str(res.q))
            return res.q

