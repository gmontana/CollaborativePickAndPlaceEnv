from gym.envs.registration import register

register(
    id='CollaborativePickAndPlace-v0',
    entry_point='pick_and_place.environment:MultiAgentPickAndPlace',
)

