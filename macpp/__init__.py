from gym.envs.registration import registry, register, make, spec
from itertools import product

register(
    id="CollaborativePickAndPlace-v0",
    entry_point="macpp.core.environment:MultiAgentPickAndPlace",
    kwargs={"width": 3, "heigth": 3, "n_agents": 2, "n_pickers": 1},
)
