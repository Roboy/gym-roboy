from gym.envs.registration import register

register(
    id='msj-control-v0',
    entry_point='gym_roboy.envs:RoboyEnv',
)