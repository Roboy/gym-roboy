from typing import Sequence
import numpy as np
from itertools import combinations
import pytest
from .. import RoboyEnv, StubROSProxy, ROSBridgeProxy
from ..robots import RobotState, MsjRobot


MSJ_ROBOT = MsjRobot()
MOCK_ROS_PROXY = StubROSProxy(robot=MSJ_ROBOT)
MOCK_ROBOY_ENV = RoboyEnv(ros_proxy=MOCK_ROS_PROXY)
constructors = [
    lambda: MOCK_ROBOY_ENV,
    pytest.param(lambda: RoboyEnv(ros_proxy=ROSBridgeProxy(robot=MSJ_ROBOT)), marks=pytest.mark.integration)
]


@pytest.fixture(
    params=constructors,
    ids=["unit-test-default", "integration"]
)
def roboy_env(request) -> RoboyEnv:
    return request.param()


def test_roboy_env_step(roboy_env):
    roboy_env.reset()
    obs, reward, done, _ = roboy_env.step(roboy_env.action_space.sample())
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool), str(type(done))


def test_roboy_env_reset(roboy_env):
    all_obs = [roboy_env.reset() for _ in range(5)]

    for obs in all_obs:
        assert np.allclose(obs[:6], 0)
        assert isinstance(obs, np.ndarray)

    for obs1, obs2 in combinations(all_obs, 2):
        assert np.allclose(obs1, obs2)


def test_roboy_env_new_goal_is_different_and_feasible(roboy_env: RoboyEnv):
    for _ in range(3):
        roboy_env._set_new_goal()
        old_goal = roboy_env._goal_state
        roboy_env._set_new_goal()
        new_goal = roboy_env._goal_state
        assert not np.allclose(old_goal.joint_angles, new_goal.joint_angles)
        assert np.all(MSJ_ROBOT.get_joint_angles_space().low <= roboy_env._goal_state.joint_angles)
        assert np.all(roboy_env._goal_state.joint_angles <= MSJ_ROBOT.get_joint_angles_space().high)


def test_roboy_env_reaching_goal_angle_delivers_maximum_reward(roboy_env: RoboyEnv):
    roboy_env.reset()
    current_joint_angles = roboy_env._last_state.joint_angles
    roboy_env._set_new_goal(goal_joint_angle=current_joint_angles)
    zero_action = np.zeros(len(roboy_env.action_space.low))
    _, reward, done, _ = roboy_env.step(zero_action)

    max_reward = roboy_env.reward_range[1]
    assert np.isclose(reward, max_reward)


def test_roboy_env_reaching_goal_joint_angle_but_moving_returns_done_equals_false(roboy_env: RoboyEnv):
    roboy_env.reset()
    current_joint_angles = roboy_env._last_state.joint_angles
    roboy_env._set_new_goal(goal_joint_angle=current_joint_angles)

    roboy_env._last_state.joint_vels = MSJ_ROBOT.get_joint_vels_space().high

    assert not roboy_env._did_complete_successfully(current_state=roboy_env._last_state,
                                                  goal_state=roboy_env._goal_state)


def test_roboy_env_joint_vel_penalty_affects_worst_possible_reward():
    env = RoboyEnv(ros_proxy=StubROSProxy(robot=MSJ_ROBOT), joint_vel_penalty=False)
    largest_distance = np.linalg.norm(2 * np.ones(MSJ_ROBOT.get_joint_angles_space().shape))
    worst_possible_reward_from_angles = -np.exp(largest_distance) - abs(env._PENALTY_FOR_TOUCHING_BOUNDARY)
    assert np.isclose(env.reward_range[0], worst_possible_reward_from_angles)

    env = RoboyEnv(ros_proxy=StubROSProxy(robot=MSJ_ROBOT), joint_vel_penalty=True)
    assert env.reward_range[0] < worst_possible_reward_from_angles


def test_roboy_env_reward_is_lower_with_joint_vel_penalty():
    new_goal_state = MsjRobot.new_random_state()
    new_goal_state.joint_vels = np.zeros_like(new_goal_state.joint_vels)

    MOCK_ROS_PROXY.forward_step_command = lambda a: new_goal_state
    env = RoboyEnv(ros_proxy=MOCK_ROS_PROXY, joint_vel_penalty=False)
    env.reset()
    env._set_new_goal(goal_joint_angle=new_goal_state.joint_angles)
    some_action = env.action_space.sample()
    _, reward_with_no_joint_vel_penalty, _, _ = env.step(action=some_action)

    env = RoboyEnv(ros_proxy=MOCK_ROS_PROXY, joint_vel_penalty=True)
    env.reset()
    env._set_new_goal(goal_joint_angle=new_goal_state.joint_angles)
    _, reward_with_joint_vel_penalty, _, _ = env.step(action=some_action)

    assert reward_with_no_joint_vel_penalty > reward_with_joint_vel_penalty


def test_roboy_env_agent_gets_bonus_when_reaching_the_goal():
    roboy_env = RoboyEnv(ros_proxy=MOCK_ROS_PROXY, is_agent_getting_bonus_for_reaching_goal=False)
    roboy_env.reset()
    reward_no_bonus = roboy_env.compute_reward(
        current_state=roboy_env._goal_state, goal_state=roboy_env._goal_state)

    roboy_env = RoboyEnv(ros_proxy=MOCK_ROS_PROXY, is_agent_getting_bonus_for_reaching_goal=True)
    roboy_env.reset()
    reward_with_bonus = roboy_env.compute_reward(
        current_state=roboy_env._goal_state, goal_state=roboy_env._goal_state)

    assert np.allclose(reward_with_bonus - reward_no_bonus, roboy_env._BONUS_FOR_REACHING_GOAL)


def test_roboy_env_render_does_nothing(roboy_env):
    roboy_env.render()


@pytest.mark.parametrize(
    "env",
    [RoboyEnv(ros_proxy=MOCK_ROS_PROXY, joint_vel_penalty=True),
     RoboyEnv(ros_proxy=MOCK_ROS_PROXY, joint_vel_penalty=False)],
    ids=["with joint_vel penalty", "no joint_vel penalty"]
)
def test_roboy_env_reward_monotonously_improves_during_approach(env: RoboyEnv):
    """
    For any random starting state, if we approach the goal at every
    step, the reward should strictly improve. This is like computing
    the numerical gradient dr/dq and checking that it is positive as
    the state approaches the goal state.

    This tests catches mistakes like reaching optimality by not moving.
    """
    np.random.seed(0)
    env.seed(0)

    num_starting_states = 40  # roughly covers many points in the space
    num_steps = 7  # distance to goal halves at every step

    goal_state = MSJ_ROBOT.new_random_zero_vel_state()

    starting_states = [MSJ_ROBOT.new_random_state() for _ in range(num_starting_states)]

    # special cases
    state_with_improvable_angles = MSJ_ROBOT.new_random_zero_vel_state()
    starting_states.append(state_with_improvable_angles)
    if env._joint_vel_penalty:
        state_with_improvable_vels = MSJ_ROBOT.new_random_zero_angle_state()
        starting_states.append(state_with_improvable_vels)

    for current_state in starting_states:
        sequence_of_rewards = []
        for _ in range(num_steps):
            reward = env.compute_reward(current_state=current_state, goal_state=goal_state)
            sequence_of_rewards.append(reward)
            current_state = RobotState.interpolate(current_state, goal_state)  # cut distance by half
        assert _strictly_increasing(sequence_of_rewards)


def test_roboy_env_maximum_episode_length():
    env = RoboyEnv(ros_proxy=MOCK_ROS_PROXY)
    env.reset()
    env.step_num = env._MAX_EPISODE_LENGTH - 1

    _, _, done, _ = env.step(np.zeros(env.action_space.shape))
    assert not done

    assert not env._did_complete_successfully(env._last_state, env._goal_state)
    _, _, done, _ = env.step(env.action_space.sample())
    assert done


def test_roboy_reset_sets_step_number_to_one():
    MOCK_ROBOY_ENV.step(MOCK_ROBOY_ENV.action_space.sample())
    assert MOCK_ROBOY_ENV.step_num is not 1

    MOCK_ROBOY_ENV.reset()
    assert MOCK_ROBOY_ENV.step_num is 1


def _strictly_increasing(sequence: Sequence[float]):
    return all(x < y for x, y in zip(sequence, sequence[1:]))
