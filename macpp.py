import argparse
import gym
import macpp
import time
import numpy as np

def game_loop(env, render=True):
    """
    Run a single game loop.
    """
    obs = env.reset()
    done = [False] * env.n_agents

    while not any(done):
        actions = env.action_space.sample()
        obs, _, done, _ = env.step(actions)
        obs = flatten_obs(obs)
        # print(obs)
        # obs = flatten_obs(obs)
        # obs = obs_to_grid(obs,(env.grid_width,env.grid_length))
        print(obs[0])
        if render:
            env.render()

        time.sleep(0.5)

def main(game_count=1, render=True):
    env = gym.make('macpp-10x10-4a-2p-3o-v0', debug_mode=True)

    for episode in range(game_count):
        game_loop(env, True)


def flatten_obs(obs):
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

def obs_to_grid(obs, grid_size):
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
    grid_width, grid_length = grid_size
    obs_all = []

    for _, agent_data in obs.items():
        grid = np.zeros((grid_width, grid_length, 6))
        x, y = agent_data['self']['position']
        grid[x, y, 3] = 1 # ID layer
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

        obs_all.append(grid)

    return obs_all



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Play the MACPP environment.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--times", type=int, default=1, help="How many times to run the game.")
    
    args = parser.parse_args()
    main(args.times, args.render)

