import numpy as np
from datetime import datetime
import os
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class QTable:

    def __init__(self, env, initial_value=0.1):
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

    # def save_table(self, filename):
    #     print(f"Saving Q-value table: {filename}.")
    #     np.save(filename, self.q_table)
    #     print(f"Number of elements in the Q table: {self.count_elements()}")
    #
    # def load_table(self, filename):
    #     print(f"Loading Q-value table: {filename}.")
    #     self.q_table = np.load(filename, allow_pickle=True).item()

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

    def epsilon_greedy_actions(self, state):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample().tolist()
        else:
            return self.greedy_actions(state)

    def greedy_actions(self, state):
        best = self.q_table.best_actions(state)
        if best is None:
            return self.env.action_space.sample().tolist()
        return best

    def act(self, state_hash, explore=False):
        # return self.epsilon_greedy_actions(state_hash) if explore else self.greedy_actions(state_hash)
        return self.epsilon_greedy_actions(state_hash) if explore else self.env.action_space.sample().tolist() 

    def learn(self, state_hash, actions, next_state_hash, rewards, done):

        # Calculate the target Q-value
        if done:
            target = rewards
        else:
            max_next_q_value = self.q_table.get_max_q_value(next_state_hash)
            target = rewards + self.discount_factor * max_next_q_value
        
        # Update the Q-value 
        current_q_value = self.q_table.get_q_value(state_hash, actions)
        updated_value = current_q_value + self.learning_rate * (target - current_q_value)
        self.q_table.set_q_value(state_hash, actions, updated_value)
        
        # Decay the exploration and learning rates
        # self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        # self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)


def game_loop(env, agent, training=False, num_episodes=1, steps_per_episode=300, render=False, create_video=False, qtable_file=None):

    print("Game loop started...")
    total_steps = []
    total_returns = []
    total_success_rate = []
    total_avg_steps = []
    total_failures = 0
    for episode in range(num_episodes):
        # print(f"Episode #{episode+1}\n")
        obs, _ = env.reset()
        obs_hash = env.obs_to_hash(obs)
        done = False
        episode_steps = 0
        episode_returns = 0
        while not done:

            # print("State before update:") 
            # env._print_state()

            # take action
            if training:
                actions = agent.act(obs_hash, explore=True)
            else:
                actions = agent.act(obs_hash, explore=False)

            # print(f"Actions: {actions}")

            # transition to new state 
            next_obs, rewards, done, _ = env.step(actions)

            # print(f'Rewards: {rewards}')

            # print("State after update:") 
            # env._print_state()

            # Hash the next state
            next_obs_hash = env.obs_to_hash(next_obs) 

            # print("State after update:") 
            # env._print_state()

            episode_returns += rewards
            episode_steps += 1

            # learn when needed
            if training:
                agent.learn(obs_hash, actions, next_obs_hash, rewards, done)
            obs = next_obs

            # render when needed
            if render:
                env.render()
                time.sleep(0.5)
            
            # check for failed episode 
            if episode_steps > steps_per_episode:
                total_failures +=1
                break
        total_steps.append(episode_steps)
        total_returns.append(episode_returns)

        # print some stats
        avg_steps = np.mean(total_steps)
        avg_return = np.mean(total_returns)
        success_rate = 1- (total_failures/(episode+1))
        print(f"Episode {episode+1}/{num_episodes}: Avg Steps: {avg_steps:.2f}, Avg Return: {avg_return:.2f}, Success rate: {success_rate:.2f}, epsilon: {agent.exploration_rate:.3f}, alpha: {agent.learning_rate:.3f}")
        total_success_rate.append(success_rate)
        total_avg_steps.append(avg_steps)

        # adjust exploration and learning rate
        agent.exploration_rate = max(agent.min_exploration, agent.exploration_rate * agent.exploration_decay)
        agent.learning_rate = max(agent.min_learning_rate, agent.learning_rate * agent.learning_rate_decay)

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episodes), total_avg_steps)
    plt.xlabel('Episode')
    plt.ylabel('Avg steps')
    plt.legend()
    plt.grid(True)
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
                      learning_rate=0.1, 
                      discount_factor=0.95, 
                      exploration_rate=1.0, 
                      exploration_decay=0.8, 
                      min_exploration=0.01, 
                      learning_rate_decay=0.9999, 
                      min_learning_rate=0.01)

    # Train the agent
    game_loop(env, agent, training=True, num_episodes=1000, steps_per_episode=300, render=False, qtable_file='qtable')
