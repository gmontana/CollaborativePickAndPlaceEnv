import gym
import numpy as np

class FlatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.env = env
        self.observation_space = None
        dim = self.n_objects * 2 * 2 + self.n_agents * 4
        self.single_space = gym.spaces.Box(np.ones(dim)*-1, np.ones(dim)*100, dtype=np.float32)
        self.observation_space = [self.single_space for _ in range(self.n_agents)]

    def observation(self, obs):
        """
        Flattens the observation dictionary into a 1D numpy array

        Args:
        obs (dict): The observation dictionary.

        Returns:
        list: List of flattened observation array as np.ndarray
        """
        flattened_obs_all = []
        num_agents = len(obs)

        # Information for each agent
        for i in range(num_agents):
            flattened_obs = []
            agent_key = f'agent_{i}'
            agent_obs = obs[agent_key]

            # Self information
            flattened_obs.extend(agent_obs['self']['position'])
            flattened_obs.append(int(agent_obs['self']['picker']))
            flattened_obs.append(agent_obs['self']['carrying_object']
                                 if agent_obs['self']['carrying_object'] is not None else -1)

            # Other agents' information
            for other_agent in agent_obs['agents']:
                flattened_obs.extend(other_agent['position'])
                flattened_obs.append(int(other_agent['picker']))
                flattened_obs.append(
                    other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

            # Objects' information
            for obj in obs[agent_key]['objects']:
                flattened_obs.extend(obj['position'])

            # Goals' information
            for goal in obs[agent_key]['goals']:
                flattened_obs.extend(goal)
            flattened_obs = np.array(flattened_obs)
            flattened_obs_all.append(flattened_obs)

        return flattened_obs_all

class GridObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = None
        single_observation_space = {}
        single_observation_space['image'] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(*env.grid_size, 6),
            dtype=np.float32)
        self.observation_space = [single_observation_space for _ in range(self.n_agents)]
    def observation(self, obs):
        '''
        Converts the observation dictionary into a 3D grid representation.

        The grid is represented as a 3D NumPy array with dimensions (grid_width, grid_length, 3),
        where the last dimension corresponds to different channels for agents, objects, and goals.
        Each cell in the grid can be either 0 or 1, indicating the absence or presence of an entity.

        Args:
        obs (dict): The observation dictionary containing information about agents, objects, and goals.
        grid_size (tuple): A tuple representing the size of the grid as (grid_width, grid_length).

        Returns:
        np.ndarray: A 3D NumPy array representing the grid.
        '''
        obs_all = []

        for _, agent_data in obs.items():
            single_obs = {}

            grid = np.zeros((*self.grid_size, 6))
            x, y = agent_data['self']['position']
            grid[x, y, 3] = 1  # ID layer
            grid[x, y, 0] = 1
            grid[x, y, 4] = 1 if agent_data['self']['carrying_object'] is not None else -1
            grid[x, y, 5] = agent_data['self']['picker']

            for other_agent in agent_data['agents']:
                x, y = other_agent['position']
                grid[x, y, 0] = 1
                grid[x, y, 4] = 1 if other_agent['carrying_object'] is not None else -1
                grid[x, y, 5] = other_agent['picker']

            for obj in agent_data['objects']:
                x, y = obj['position']
                grid[x, y, 1] = 1

            for goal in agent_data['goals']:
                x, y = goal
                grid[x, y, 2] = 1

            single_obs['image'] = grid
            obs_all.append(single_obs)

        return (obs_all)
