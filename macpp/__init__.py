from gym.envs.registration import registry, register, make, spec
from itertools import product


grid_sizes = [(3, 3), (5, 5), (10, 10)]

# Register the environments
for grid_size in grid_sizes:
    for n_agents in [2, 4]:
        for n_pickers in [1, 2]:
            for n_objects in [1, 2, 3]:
                env_name = f"macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}-{n_pickers}-{n_objects}-v0"
                register(
                    id=env_name,
                    entry_point='macpp.envs:MultiAgentPickAndPlace',
                    kwargs={
                        'grid_size': grid_size,
                        'n_agents': n_agents,
                        'n_pickers': n_pickers,
                        'n_objects': n_objects
                    }
               )
