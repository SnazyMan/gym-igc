from gym.envs.registration import register

register(
    id='igc-v0',
    entry_point='gym_igc.envs:igcEnv',
)
