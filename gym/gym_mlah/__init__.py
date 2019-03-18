from gym.envs.registration import register

register(
        'GridWorld-v0',
        entry_point='gym_mlah.envs.custom:GridWorld',
)

