from gym.envs.registration import register

register(
    id='franka_pole-v0',
    entry_point='gym_franka_pole.envs:franka_pole_Env',
)
