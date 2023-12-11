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
        obs, reward, done, _ = env.step(actions)
        # print(obs)
        obs = flatten_obs(obs)
        # print(len(obs[0]),len(obs[1]))
        if render:
            env.render()
        print(reward)
        time.sleep(0.5)

def main(game_count=1, render=True):
    env = gym.make('macpp-5x5-2a-1p-3o-sparse-v0', debug_mode=False)

    for episode in range(game_count):
        game_loop(env, render=False)


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Play the MACPP environment.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--times", type=int, default=1, help="How many times to run the game.")
    
    args = parser.parse_args()
    main(args.times, args.render)

