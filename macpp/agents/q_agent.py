import numpy as np
from datetime import datetime
import os
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

class QTable:

    def __init__(self, env, initial_value=0.0):
        self.env = env
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.initial_value = initial_value

    def get_q_value(self, state_hash, actions):
        return self.q_table.get(state_hash, {}).get(tuple(actions), self.initial_value)

    def set_q_value(self, state_hash, actions, value):
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        self.q_table[state_hash][tuple(actions)] = value

    def get_max_q_value(self, state_hash):
        if state_hash not in self.q_table:
            return self.initial_value
        return max(self.q_table[state_hash].values())

    def best_actions(self, state_hash):
        if state_hash not in self.q_table:
            return None
        max_q_value = self.get_max_q_value(state_hash)
        best_acts = [act for act, q_value in self.q_table[state_hash].items() if q_value == max_q_value]
        return list(best_acts[0]) if best_acts else None

    def save_q_table(self, filename):
        print(f"Saving Q-value table: {filename}.")
        np.save(filename, self.q_table)
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def load_q_table(self, filename):
        print(f"Loading Q-value table: {filename}.")
        self.q_table = np.load(filename, allow_pickle=True).item()

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99999, min_exploration=0.03, learning_rate_decay=0.99999, min_learning_rate=0.01):
        self.env = env
        self.n_agents = env.n_agents
        self.action_space_size = env.n_agents
        self.q_table = QTable(self.env)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.state_visits = {}  
        self.state_action_visits = {}  

    def epsilon_greedy_actions(self, obs):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample().tolist()
        else:
            return self.greedy_actions(obs)

    def greedy_actions(self, obs):
        best = self.q_table.best_actions(obs)
        if best is None:
            return self.env.action_space.sample().tolist()
        return best

    def ucb_action_selection(self, obs_hash):
        c = 0.9
        max_ucb = float('-inf')
        best_action = None

        # Generate all possible action combinations for the MultiDiscrete action space
        action_combinations = [list(x) for x in np.ndindex(*self.env.action_space.nvec)]

        for action in action_combinations:
            q_value = self.q_table.get_q_value(obs_hash, action)
            state_count = self.state_visits.get(obs_hash, 1)  # Avoid division by zero
            state_action_key = (obs_hash, tuple(action))
            state_action_count = self.state_action_visits.get(state_action_key, 1)
            ucb_value = q_value + c * np.sqrt(np.log(state_count) / state_action_count)

            if ucb_value > max_ucb:
                max_ucb = ucb_value
                best_action = action

        return best_action


    def act(self, obs_hash, explore=False):
        if explore:
            return self.ucb_action_selection(obs_hash)
        else:
            return self.greedy_actions(obs_hash)

    def learn(self, obs_hash, actions, next_state_hash, rewards, done):

        # Calculate the target Q-value
        if done:
            target = rewards
        else:
            max_next_q_value = self.q_table.get_max_q_value(next_state_hash)
            target = rewards + self.discount_factor * max_next_q_value
        
        # Update the Q-value 
        current_q_value = self.q_table.get_q_value(obs_hash, actions)
        updated_value = current_q_value + self.learning_rate * (target - current_q_value)
        self.q_table.set_q_value(obs_hash, actions, updated_value)
        
        # Decay the exploration and learning rates
        # self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        # self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)


def game_loop(env, agent, training=False, num_episodes=1, msx_steps_per_episode=300, render=False, create_video=False, qtable_file=None):

    print("Game loop started...")
    total_steps = []
    total_returns = []
    total_success_rate = []
    total_avg_steps = []
    total_avg_returns = []
    total_failures = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs_hash = env.obs_to_hash(obs)
        done = False
        episode_steps = 0
        episode_returns = 0

        while not done:
            if training:
                actions = agent.act(obs_hash, explore=True)
            else:
                actions = agent.act(obs_hash, explore=False)

            # Update the state visit count
            agent.state_visits[obs_hash] = agent.state_visits.get(obs_hash, 0) + 1

            # Update state-action visit count
            state_action_key = (obs_hash, tuple(actions))
            agent.state_action_visits[state_action_key] = agent.state_action_visits.get(state_action_key, 0) + 1

            next_obs, rewards, done, _ = env.step(actions)
            next_obs_hash = env.obs_to_hash(next_obs)

            episode_returns += rewards
            episode_steps += 1

            if training:
                agent.learn(obs_hash, actions, next_obs_hash, rewards, done)
            obs = next_obs

            if render:
                env.render()
                time.sleep(0.5)
            
            if episode_steps > msx_steps_per_episode:
                total_failures +=1
                break
        total_steps.append(episode_steps)
        total_returns.append(episode_returns)

        avg_steps = np.mean(total_steps)
        avg_return = np.mean(total_returns)
        success_rate = 1- (total_failures/(episode+1))
        total_success_rate.append(success_rate)
        total_avg_steps.append(avg_steps)
        total_avg_returns.append(avg_return)

        agent.exploration_rate = max(agent.min_exploration, agent.exploration_rate * agent.exploration_decay)
        agent.learning_rate = max(agent.min_learning_rate, agent.learning_rate * agent.learning_rate_decay)

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}: Avg Steps: {avg_steps:.2f}, Avg Return: {avg_return:.2f}, Success rate: {success_rate:.2f}, epsilon: {agent.exploration_rate:.3f}, alpha: {agent.learning_rate:.3f}")

    # Create the main figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first metric on the primary y-axis
    ax1.plot(range(num_episodes), total_avg_returns, 'b-', label='Avg Return')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg Return', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create the second y-axis and plot the second metric on it
    ax2 = ax1.twinx()
    ax2.plot(range(num_episodes), total_avg_steps, 'r-', label='Avg Steps')
    ax2.set_ylabel('Avg Steps', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')

    plt.show()

    # create a video when needed 
    if create_video:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=script_dir+f"/videos/episode_{timestamp}.mp4"
        print(f"Saving movie in {filename}.")
        env.save_video(filename)

    # if training and qtable_file is not None:
    #     agent.q_table.save_table(qtable_file)

if __name__ == "__main__":

    from macpp.core.environment import MACPPEnv

    # Set up the environment
    env = MACPPEnv(
        grid_size=(3, 3), n_agents=2, n_pickers=1, n_objects=1, cell_size=300, debug_mode=False
    )

    # Set up the Q agent
    agent = QLearning(env, 
                      learning_rate=0.3,
                      discount_factor=0.98, 
                      exploration_rate=1.0, 
                      exploration_decay=0.995, 
                      min_exploration=0.01, 
                      learning_rate_decay=0.999, 
                      min_learning_rate=0.01)

    # Train the agent
    game_loop(env, agent, training=True, num_episodes=100000, msx_steps_per_episode=300, render=False, qtable_file='qtable')
