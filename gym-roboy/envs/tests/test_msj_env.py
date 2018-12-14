from .. import MsjEnv

env = MsjEnv()
random_action = env.action_space.sample()


def test_msj_env_step():
    env.step(random_action)
