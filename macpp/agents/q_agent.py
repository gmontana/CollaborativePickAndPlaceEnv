import numpy as np
from datetime import datetime
import os
import random
import json
import time
from gym.spaces import Tuple, Discrete, Dict
import itertools
from collections import defaultdict

class QTable:

    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def get_q_value(self, state, actions):
        """Retrieve the Q-value for a given state and actions."""
        agent_actions = tuple(actions)
        if agent_actions not in self.q_table[state]:
            # Initialize to a small random value between -0.01 and 0.01
            self.q_table[state][agent_actions] = np.random.uniform(-0.01, 0.01)
        return self.q_table[state][agent_actions]

    def set_q_value(self, state, actions, value):
        """Set the Q-value for a given state and actions."""
        agent_actions = tuple(actions)
        self.q_table[state][agent_actions] = value

    def get_max_q_value(self, state):
        """Retrieve the maximum Q-value for a given state."""
        if state not in self.q_table:
            return 0.0
        all_q_values = [v for action, v in self.q_table[state].items()]
        return max(all_q_values, default=0.0)

    def best_actions(self, state):
        """Retrieve the actions that yield the maximum Q-value for a given state."""
        if state not in self.q_table:
            return None
        max_q_value = self.get_max_q_value(state)
        best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
        return list(best_actions[0]) if best_actions else None

    def initialise(self, state):
        """Ensure the state is initialized in the Q-table."""
        if state not in self.q_table:
            self.q_table[state] = defaultdict(lambda: defaultdict(float))


    # def save_table(self, filename):
    #     print(f"Saving Q-value table: {filename}.")
    #     np.save(filename, self.q_table)
    #     print(f"Number of elements in the Q table: {self.count_elements()}")
    #
    # def load_table(self, filename):
    #     print(f"Loading Q-value table: {filename}.")
    #     self.q_table = np.load(filename, allow_pickle=True).item()
    #     print(f"Number of elements in the Q table: {self.count_elements()}")

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99, min_exploration=0.02, learning_rate_decay=0.995, min_learning_rate=0.01, max_steps_per_episode=50):
        self.env = env
        self.n_agents = env.n_agents
        self.action_space_size = env.n_agents
        self.q_table = QTable()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.max_steps_per_episode = max_steps_per_episode

    def epsilon_greedy_actions(self, state):
        state_str = json.dumps(state)
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample().tolist()
        else:
            best_actions_list = self.q_table.best_actions(state_str)
            if best_actions_list:
                return random.choice(best_actions_list)
            return self.env.action_space.sample().tolist()

    def greedy_actions(self, state):
        state_str = json.dumps(state)
        best_actions_list = self.q_table.best_actions(state_str)
        if best_actions_list:
            return random.choice(best_actions_list)
        return self.env.action_space.sample().tolist()

    def act(self, state, explore=False):
        state_str = json.dumps(state)
        return self.epsilon_greedy_actions(state) if explore else self.greedy_actions(state)

    def learn(self, state, actions, next_state, rewards, done):
        state_str = json.dumps(state)
        next_state_str = json.dumps(next_state)

        total_reward = sum(rewards)
        
        # Calculate the target Q-value
        if done:
            target = total_reward
        else:
            max_next_q_value = self.q_table.get_max_q_value(next_state_str)
            target = total_reward + self.discount_factor * max_next_q_value
        
        # Update the Q-value using the Q-learning formula
        current_q_value = self.q_table.get_q_value(state_str, actions)
        updated_value = current_q_value + self.learning_rate * (target - current_q_value)
        self.q_table.set_q_value(state_str, actions, updated_value)
        
        # Decay the exploration and learning rates
        # self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        # self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)


def game_loop(env, agent, training=False, num_episodes=1, steps_per_episode=50, render=False, create_video=False, qtable_file=None):

    print("Game loop started...")
    total_steps = []
    total_returns = []
    for episode in range(num_episodes):
        # print(f"Episode #{episode+1}\n")
        state = env.reset()
        # print(env._print_state())
        done = False
        episode_steps = 0
        episode_returns = 0
        while not done:
            print("State before update:") 
            env._print_state()

            # take action
            if training:
                actions = agent.act(state, explore=True)
            else:
                actions = agent.act(state, explore=False)

            # actions = random.sample(range(6), 2) 

            print(f"Actions: {actions}")
            # execute action

            next_state, rewards, done, _ = env.step(actions)
            
            # print("State after update:") 
            # env._print_state()

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

        agent.exploration_rate = max(agent.min_exploration, agent.exploration_rate * agent.exploration_decay)
        agent.learning_rate = max(agent.min_learning_rate, agent.learning_rate * agent.learning_rate_decay)

    # create a video when needed 
    if create_video:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=script_dir+f"/videos/episode_{timestamp}.mp4"
        print(f"Saving movie in {filename}.")
        env.save_video(filename)

    # if training and qtable_file is not None:
    #     agent.q_table.save_table(qtable_file)


    return total_steps, total_returns

if __name__ == "__main__":

    from macpp.core.environment import MultiAgentPickAndPlace

    env = MultiAgentPickAndPlace(
        width=3, length=3, n_agents=2, n_pickers=1, cell_size=300, debug_mode=False
    )
    agent = QLearning(env)
    total_steps, total_returns = game_loop(env, agent, True, 20, 10, render=False, qtable_file='qtable')
    # total_steps, total_returns = game_loop(env, agent, False, 10, create_video=True, qtable_file='qtable')
    # print(env.action_space)
    # print(env._get_action_space_size())
    # print(env.state_space)
    # print(env._get_state_space_size())

