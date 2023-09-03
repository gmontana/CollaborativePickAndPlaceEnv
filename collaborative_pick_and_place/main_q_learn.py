import random
from macpp import MultiAgentPickAndPlace
import numpy as np
from sacred import Experiment

ex = Experiment('q_learning_experiment')

class QTable:
    def __init__(self, n_agents, action_space):
        self.q_table = {}
        self.n_agents = n_agents
        self.action_space = action_space

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((len(self.action_space),) * self.n_agents)
        return self.q_table[state]

    def update(self, state, actions, value):
        current_q_values = self.get_q_values(state)
        action_indices = tuple(self.action_space.index(action) for action in actions)
        current_q_values[action_indices] = value


class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.q_table = QTable(n_agents=env.n_agents, action_space=env.get_action_space())
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

    def choose_actions(self, state_hash):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return [np.random.choice(self.q_table.action_space) for _ in range(self.q_table.n_agents)]
        q_values = self.q_table.get_q_values(state_hash)
        actions_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
        return [self.q_table.action_space[index] for index in actions_indices]

    def train(self, episodes, max_steps_per_episode):
        rewards_all_episodes = []

        for episode in range(episodes):
            state = self.env.reset()
            state_hash = self.env.get_hashed_state()
            
            done = False
            rewards_current_episode = 0

            for step in range(max_steps_per_episode):
                actions = self.choose_actions(state_hash)
                next_state, rewards, done = self.env.step(actions)
                next_state_hash = self.env.get_hashed_state()

                total_reward = sum(rewards)
                current_q_values = self.q_table.get_q_values(state_hash)
                next_q_values = self.q_table.get_q_values(next_state_hash)
                action_indices = tuple(self.q_table.action_space.index(action) for action in actions)
                max_next_q_value = np.max(next_q_values)
                
                if done:
                    target = total_reward
                else:
                    target = total_reward + self.discount_factor * max_next_q_value
                
                updated_value = current_q_values[action_indices] + self.learning_rate * (target - current_q_values[action_indices])
                self.q_table.update(state_hash, actions, updated_value)

                state_hash = next_state_hash
                rewards_current_episode += total_reward

                if done:
                    break

            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
            rewards_all_episodes.append(rewards_current_episode)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_all_episodes[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward}")

        return self.q_table, rewards_all_episodes


@ex.config
def cfg():

    # Environment parameters
    env_width = 4
    env_length = 4
    env_n_agents = 2
    env_n_pickers = 1
    env_enable_rendering = False

   # Q-learning parameters
    episodes = 2000
    max_steps_per_episode = 300
    learning_rate = 0.3
    discount_factor = 0.99
    exploration_rate = 0.9
    exploration_decay = 0.995
    min_exploration = 0.01

@ex.main
def run_experiment(episodes, max_steps_per_episode, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration, env_width, env_length, env_n_agents, env_n_pickers, env_enable_rendering):

    # initialise env
    env = MultiAgentPickAndPlace(width=env_width, length=env_length, n_agents=env_n_agents, n_pickers=env_n_pickers, enable_rendering=env_enable_rendering)

    # train algo
    q_learning = QLearning(env, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration)
    q_table, rewards_all_episodes = q_learning.train(episodes, max_steps_per_episode)

    # return stats
    # avg_rewards = np.mean(rewards_all_episodes)
    # return {"average_rewards": avg_rewards, "all_rewards": rewards_all_episodes}

if __name__ == "__main__":
    ex.run_commandline()

