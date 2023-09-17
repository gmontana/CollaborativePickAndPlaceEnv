import numpy as np
from datetime import datetime
import os
import random


class QLearningTable:
    def __init__(self, n_agents, action_space):
        self.q_table = {}
        self.n_agents = n_agents
        self.action_space = action_space

    def initialise(self, state):
        ''' 
        Initialising the Q table with small values may help exploration early on
        '''
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(-0.01, 0.01, (len(self.action_space),) * self.n_agents)
        return self.q_table[state]

    def update(self, state, actions, value):
        current_q_values = self.initialise(state)
        action_indices = tuple(self.action_space.index(action) for action in actions)
        current_q_values[action_indices] = value

    def count_elements(self):
        return sum([np.prod(v.shape) for v in self.q_table.values()])

    def get_max_q_value(self, state):
        current_q_values = self.initialise(state)
        return np.max(current_q_values)

    def save_q_table(self, filename):
        print(f"Saving Q-value table: {filename}.")
        np.save(filename, self.q_table)
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def load_q_table(self, filename):
        print(f"Loading Q-value table: {filename}.")
        self.q_table = np.load(filename, allow_pickle=True).item()
        print(f"Number of elements in the Q table: {self.count_elements()}")


class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01, learning_rate_decay=0.995, min_learning_rate=0.01, max_steps_per_episode=50):
        self.env = env
        self.q_table = QLearningTable(n_agents=env.n_agents, action_space=env.get_action_space())
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
            return [np.random.choice(self.q_table.action_space) for _ in range(self.q_table.n_agents)]
        else:
            return self.greedy_actions(state)

    def greedy_actions(self, state):
        q_values = self.q_table.initialise(state)
        actions_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
        return [self.q_table.action_space[index] for index in actions_indices]

    def execute(self, max_num_steps, save_video=False):

        # Get initial state
        state = self.env.reset()
        # state = self.env.get_hashed_state()

        done = False
        success = False
        total_steps = 0
        total_return = 0

        print("-----------------------")
        print("NEW EPISODE STARTS HERE")
        print(f"Initial state: {state}")
        print("-----------------------")

        while not done and total_steps < max_num_steps:  
            total_steps += 1

            print("-----------------------")
            actions = self.greedy_actions(state)
            print(f"Agents' actions: {actions}")
            next_state , rewards, done = self.env.step(actions)
            # next_state_hash = self.env.get_hashed_state()
            print(f"Next state: {next_state}")
            print("-----------------------")

            total_reward = sum(rewards)
            state = next_state
            total_return += total_reward

            if done: 
                print(f"EPISODE ENDED WITH SUCCESS")
                success = True

        if save_video:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename=script_dir+f"/videos/episode_{timestamp}.mp4"
            print(f"Saving movie in {filename}.")
            self.env.save_video(filename)

        return success, total_return, total_steps

    def train(self, state):

        # Get actions, next state and rewards
        actions = self.epsilon_greedy_actions(state)
        next_state, rewards, done = self.env.step(actions)

        # Update Q-values
        total_reward = sum(rewards)
        # current_q_values = self.q_table.initialise(state)
        # next_q_values = self.q_table.initialise(next_state)
        action_indices = tuple(self.q_table.action_space.index(action) for action in actions)
        max_next_q_value = self.get_max_q_value(next_state)
        if done:
            target = total_reward
        else:
            target = total_reward + self.discount_factor * max_next_q_value
        updated_value = current_q_values[action_indices] + self.learning_rate * (target - current_q_values[action_indices])
        self.q_table.update(state, actions, updated_value)

        # Decay exploration and learning rates
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)

        return total_reward, done, next_state
