from gymnasium.envs.registration import register

register(
    id='pvz-env-v2',
    entry_point='gympvz.gym_pvz.envs:PVZEnv_V2'
)