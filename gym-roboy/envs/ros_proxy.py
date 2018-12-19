import numpy as np
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


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
        self.new_goal_publisher = self.node.create_publisher(
            msg_type=PoseStamped, topic="/robot_state_target")

    def _log_robot_state(self, robot_state):
        q = robot_state.q
        qd = robot_state.qdot
        q_str = str(q).strip('[]')
        qd_str = str(qd).strip('[]')
        self.node.get_logger().info("joint angles: %s" % q_str)
        self.node.get_logger().info("joint velocity: %s" % qd_str)

    def forward_reset_command(self):
        req = GymStep.Request()
        res = GymStep.Response()
        req.step_size = self._step_size
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)
        res = future.result()
        if future.result() is not None:
            self._log_robot_state(res)
        return self._make_robot_state(service_response=res)

    @staticmethod
    def _make_robot_state(service_response) -> MsjRobotState:
        return MsjRobotState(joint_angle=service_response.q[3:],
                             joint_vel=service_response.qdot[3:])

    def forward_step_command(self, action):

        while not self.step_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')

        req = GymStep.Request()
        res = GymStep.Response()
        req.set_points = action
        req.step_size = self._step_size
        future = self.step_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)
        res = future.result()
        if future.result() is not None:
            self._log_robot_state(res)
            #self.node.get_logger().info("feasible: " + str(res.feasible))
            if not res.feasible:
                return self.forward_reset_command()
        return self._make_robot_state(res)

    def read_state(self):
        while not self.step_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')
        req = GymStep.Request()
        future = self.step_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)

        res = future.result()
        return self._make_robot_state(res)

    def forward_new_goal(self, goal_joint_angle):
        assert len(goal_joint_angle) == 3

        point = Point()
        point.x = goal_joint_angle[0]
        point.y = goal_joint_angle[1]
        point.z = goal_joint_angle[2]

        quaternion = Quaternion()
        quaternion.x = 0.01
        quaternion.z = 0.01
        quaternion.y = 0.01
        quaternion.w = 0.01

        pose = Pose()
        pose.position = point
        pose.orientation = quaternion

        msg = PoseStamped()
        msg.pose = pose

        self.node.get_logger().info("Publishing: " + str(msg.pose))
        self.new_goal_publisher.publish(msg=msg)
