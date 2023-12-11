from gym.envs.registration import registry, register, make, spec
from itertools import product


grid_sizes = [(3, 3), (5, 5), (10, 10), (15, 15), (20, 20)]
n_agents_values = [2, 4]
n_pickers_values = [1, 2, 3]
n_objects_values = [1, 2, 3, 4]
sparse_values = [True, False]


# Register the environments
for grid_size, n_agents, n_pickers, n_objects, sparse in product(grid_sizes,
                                                                n_agents_values,
                                                                n_pickers_values,
                                                                n_objects_values,
                                                                sparse_values):
    env_name = f"macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}a-{n_pickers}p-{n_objects}o-{'sparse-' if sparse else ''}v0"
    register(
        id=env_name,
        entry_point='macpp.core.environment:MACPPEnv',
        kwargs={
            'grid_size': grid_size,
            'n_agents': n_agents,
            'n_pickers': n_pickers,
            'n_objects': n_objects,
            'sparse_reward': sparse,
        }
    )

