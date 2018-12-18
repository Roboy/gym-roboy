import numpy as np
import rclpy
from roboy_simulation_msgs.srv import GymStep
from roboy_simulation_msgs.srv import GymReset


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
        pass

class MsjROSBridgeProxy(MsjROSProxy):

    def __init__(self):
        try:
            rclpy.init()
        except:
            pass
        self.node = rclpy.create_node('gym_client')
        self.step_cli = self.node.create_client(GymStep, 'gym_step')
        self.reset_cli = self.node.create_client(GymReset, 'gym_reset')
        self.step_size = 1.0;
    
    def forward_reset_command(self):

        while not self.reset_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')

        req = GymReset.Request()

        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)
        if future.result() is not None:
            q = future.result().q
            qd = future.result().qdot
            q_str = str(q).strip('[]')
            qd_str = str(qd).strip('[]')
            self.node.get_logger().info("Reset, joint angles: %s" % q_str)
            self.node.get_logger().info("Reset, joint velocity: %s" % qd_str)

        robot_state = future.result()
        return MsjRobotState(joint_angle=robot_state.q, joint_vel=robot_state.qdot)

    def forward_step_command(self, action):

        while not self.step_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')

        step_size = 1.0
        req = GymStep.Request()
        res = GymStep.Response()
        req.set_points = action
        req.step_size = self.step_size
        future = self.step_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)
        if future.result() is not None:
            q = future.result().q
            qd = future.result().qdot
            q_str = str(q).strip('[]')
            qd_str = str(qd).strip('[]')
            self.node.get_logger().info("Step, joint angles: %s" % q_str)
            self.node.get_logger().info("Step, joint velocity: %s" % qd_str)
            
        robot_state = future.result()
        return MsjRobotState(joint_angle=robot_state.q, joint_vel=robot_state.qdot)


    def read_state(self):
        while not self.step_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting...')
        req = GymStep.Request()
        future = self.step_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node,future)
        if future.result() is not None:
            q = future.result().q
            qd = future.result().qdot
            q_str = str(q).strip('[]')
            qd_str = str(qd).strip('[]')
            self.node.get_logger().info("Observation, joint angles: %s" % q_str)
            self.node.get_logger().info("Observation, joint velocity: %s" % qd_str)



        robot_state = future.result()
        return MsjRobotState(joint_angle=robot_state.q, joint_vel=robot_state.qdot)