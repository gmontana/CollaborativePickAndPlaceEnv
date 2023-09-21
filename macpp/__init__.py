from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(3, 5)  # Grid sizes
agents = range(2, 5)  # Number of agents
objects = range(1, 5)  # Number of objects

for s, a, o in product(sizes, agents, objects):
    pickers = range(1, a)  # Dynamic range for pickers based on n_agents
    for p in pickers:
        register(
            id="MACPP-{0}x{0}-{1}a-{2}o-{3}p-v0".format(s, a, o, p),
            entry_point="macpp.core.environment:MultiAgentPickAndPlace",
            kwargs={
                "width": s,
                "length": s,
                "n_agents": a,
                "n_pickers": p,
                "n_objects": o
            }
        )

