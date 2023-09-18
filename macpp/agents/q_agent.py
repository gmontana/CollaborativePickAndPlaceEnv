import numpy as np
from datetime import datetime
import os
import random
import json
import time
from gym.spaces import Tuple, Discrete, Dict
import itertools

class QLearningTable:
    def __init__(self, action_space):
        self.q_table = {}
        self.n_agents= action_space
        self.action_size = 10

    def initialise(self, state):
        state_str = json.dumps(state)  
        if state_str not in self.q_table:
            self.q_table[state_str] = np.random.uniform(-0.01, 0.01, [self.action_size] * self.n_agents)
        return self.q_table[state_str]

    def update(self, state):
        state_str = json.dumps(state)
        current_q_values = self.initialise(state_str)

    def count_elements(self):
        return sum([np.prod(v.shape) for v in self.q_table.values()])

    def get_max_q_value(self, state):
        state_str = json.dumps(state)
        current_q_values = self.initialise(state_str)
        return np.max(current_q_values)

    def save_table(self, filename):
        print(f"Saving Q-value table: {filename}.")
        np.save(filename, self.q_table)
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def load_table(self, filename):
        print(f"Loading Q-value table: {filename}.")
        self.q_table = np.load(filename, allow_pickle=True).item()
        print(f"Number of elements in the Q table: {self.count_elements()}")

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01, learning_rate_decay=0.995, min_learning_rate=0.01, max_steps_per_episode=50):
        self.env = env
        self.n_agents = env.n_agents
        self.action_space_size = _get_action_space_size(env.action_space)
        self.q_table = QLearningTable(self.action_space_size)
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.max_steps_per_episode = max_steps_per_episode

    def epsilon_greedy_actions(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return [self.env.action_space.sample() for _ in range(self.n_agents)]
        else:
            return self.greedy_actions(state)

    def greedy_actions(self, state):
        q_values = self.q_table.initialise(state)
        combined_action = np.unravel_index(np.argmax(q_values), q_values.shape)
        adjusted_action = [action + 1 for action in combined_action]
        return adjusted_action

    def act(self, state, explore=False):
        if explore:
            return self.epsilon_greedy_actions(state)
        else:
            return self.greedy_actions(state)

    def learn(self, state, actions, next_state, rewards, done):
        total_reward = sum(rewards)
        max_next_q_value = self.q_table.get_max_q_value(next_state)
        
        if done:
            target = total_reward
        else:
            target = total_reward + self.discount_factor * max_next_q_value
        
        current_q_values = self.q_table.initialise(state)
        action_indices = tuple(actions)
        updated_value = current_q_values[action_indices] + self.learning_rate * (target - current_q_values[action_indices])
        self.q_table.update(state, actions, updated_value)
        
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
        
        return total_reward, done, next_state

def _get_action_space_size(action_space):
        if isinstance(action_space, Tuple):
            discrete_sizes = [space.n for space in action_space if isinstance(space, Discrete)]
            if discrete_sizes:
                return np.prod(discrete_sizes)
        return None  

def game_loop(env, agent, training=False, num_episodes=1, steps_per_episode=50, render=False, create_video=False, qtable_file=None):

    print("Game loop started...")
    total_steps = []
    total_returns = []
    for episode in range(num_episodes):
        print(f"Episode #{episode+1}\n")
        state = env.reset()
        print(env._print_state())
        done = False
        episode_steps = 0
        episode_returns = 0
        while not done:
            # print(f"Current state: {env._print_state()}")
            # take action
            if training:
                actions = agent.act(state, explore=True)
            else:
                actions = agent.act(state, explore=False)
            # execute action
            next_state, rewards, done, _ = env.step(actions)
            episode_returns += sum(rewards)
            episode_steps += 1
            # learn when needed
            if training:
                agent.learn(state, actions, next_state, rewards, done)
            # print(f"Actions: {actions}")
            # print(f"Next state: {env._print_state()}")
            state = next_state
            if render:
                env.render()
                time.sleep(0.5)
            if episode_steps > steps_per_episode:
                break
        total_steps.append(episode_steps)
        total_returns.append(episode_returns)

        # print some stats
        avg_steps = np.mean(total_steps)
        avg_return = np.mean(total_returns)
        print(f"Episode {episode+1}/{num_episodes}: Average Steps: {avg_steps:.2f}, Average Return: {avg_return:.2f}")

    # create a video when needed 
    if create_video:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=script_dir+f"/videos/episode_{timestamp}.mp4"
        print(f"Saving movie in {filename}.")
        env.save_video(filename)
    if training and qtable_file is not None:
        agent.q_table.save_table(qtable_file)


    return total_steps, total_returns

if __name__ == "__main__":

    from macpp.core.environment import MultiAgentPickAndPlace

    env = MultiAgentPickAndPlace(
        width=3, length=3, n_agents=2, n_pickers=1, cell_size=300, debug_mode=False
    )
    agent = QLearning(env)

    print(env.action_space)
    print(env._get_action_space_size())
    print(env.state_space)
    print(env._get_state_space_size())

    # total_steps, total_returns = game_loop(env, agent, True, 1000, 300, render=False, qtable_file='qtable')
    # total_steps, total_returns = game_loop(env, agent, False, 10, create_video=True, qtable_file='qtable')
