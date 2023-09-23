import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from exploration import EpsilonGreedy
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env, exploration_strategy, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64, tau=0.1):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = DQNNetwork(self.state_size, self.action_size).to(device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.memory = ExperienceReplay(buffer_size)
        self.exploration_strategy = exploration_strategy

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.exploration_strategy.select_action(self, state):
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.network(state)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def decay_epsilon(self):
        self.exploration_strategy.decay_exploration_rate()


def game_loop(env, agent, training=True, num_episodes=10000, max_steps_per_episode=300, render=False, model_file='dqn_model'):
    total_rewards = []
    all_losses = []
    failure_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            if training:
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.train()
                all_losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

            if step >= max_steps_per_episode:
                failure_count +=1

        total_rewards.append(episode_reward)

        # Decay epsilon for exploration-exploitation trade-off
        if training:
            agent.decay_epsilon()

        # Periodically update the target network
        if training and episode % 100 == 0:
            agent.update_target_network()

        # Logging
        if episode % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            success_rate = 1 - (failure_count / 100)
            avg_loss = np.mean(all_losses[-100:])
            print(f"Episode {episode}/{num_episodes}: Avg Reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.exploration_strategy.epsilon:.2f}")

        # Save the model periodically
        if training and episode % 1000 == 0:
            torch.save(agent.network.state_dict(), model_file + f"_{episode}.pth")

    agent.exploration_strategy.decay()

    if training:
        torch.save(agent.network.state_dict(), model_file + "_final.pth")


if __name__ == "__main__":

    from macpp.core.environment import MACPPEnv

    # Set up the environment
    env = MACPPEnv(
        grid_size=(3, 3), n_agents=2, n_pickers=1, n_objects=1, cell_size=300, debug_mode=False
    )

    # Set up exploration strategy
    epsilon_greedy_strategy = EpsilonGreedy(exploration_rate=1.0, min_exploration=0.02, exploration_decay=0.99)

    # Set up the agent
    agent = DQNAgent(env, epsilon_greedy_strategy,learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64, tau=0.1)

    # Train the agent
    game_loop(env, agent, training=True, num_episodes=5000, max_steps_per_episode=500, render=False, model_file='dqn_model')

