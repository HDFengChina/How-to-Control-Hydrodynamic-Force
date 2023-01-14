from gym.envs.registration import register

register(
    id='foil-v0',
    entry_point="env.flow_field_env:foil_env"
)