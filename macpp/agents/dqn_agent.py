import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing import Dict, Any
from abc import ABC, abstractmethod


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def flatten_obs(obs):
    flattened = []
    for agent, agent_obs in obs.items():
        # Agent's own position
        flattened.extend(agent_obs['self']['position'])
        # Picker status (1 if True, 0 if False)
        flattened.append(1 if agent_obs['self']['picker'] else 0)
        # Carrying object status (1 if carrying, 0 if not)
        flattened.append(1 if agent_obs['self']['carrying_object'] else 0)
        
        # Other agents' information
        for other_agent in agent_obs['agents']:
            flattened.extend(other_agent['position'])
            flattened.append(1 if other_agent['picker'] else 0)
            flattened.append(1 if other_agent['carrying_object'] else 0)
        
        # Objects' information
        for obj in agent_obs['objects']:
            flattened.extend(obj['position'])
            flattened.append(obj['id'])
        
        # Goals' information
        for goal in agent_obs['goals']:
            flattened.extend(goal)

        # Concatene the agent's obs along the 1D vector
        flattened_obs = np.array(flattened)
    
    return flattened_obs

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values!")

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    '''
    This DQN implementation treats the multi-agent problem as a single-agent problem by flattening the observations and encoding the joint actions. 
    The Q-value network outputs Q-values for all possible joint actions, and the action with the highest Q-value is selected.
    '''
    def __init__(self, env, exploration_strategy, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64, tau=0.1):
        self.env = env
        agent_obs_len = 4 + (4 * (self.env.n_agents - 1)) + (3 * self.env.n_objects) + (2 * self.env.n_objects)
        self.obs_size = agent_obs_len * self.env.n_agents
        self.action_size = np.prod(env.action_space.nvec)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = 6 
        self.network = DQNNetwork(self.obs_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.obs_size, self.action_size).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.memory = ExperienceReplay(buffer_size)
        self.exploration_strategy = exploration_strategy

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size


    def select_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)  
        q_values = self.network(state_tensor)
        
        if training and random.random() < self.exploration_strategy.epsilon:
            # Exploration: Randomly choose actions for each agent
            actions = [random.choice(range(self.n_actions)) for _ in range(self.env.n_agents)]
        else:
            # Exploitation: Choose best actions based on Q-values
            actions = q_values.argmax(dim=2).squeeze(0).tolist()
        
        return actions

    def encode_joint_actions(self, actions):
        joint_actions = torch.zeros(actions.size(0), dtype=torch.long, device=self.device)
        for agent_idx in range(self.env.n_agents):
            joint_actions = joint_actions * self.n_actions + actions[:, agent_idx]
        return joint_actions

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)

        joint_actions = self.encode_joint_actions(actions)

        rewards = torch.tensor(rewards).to(self.device).view(-1, self.env.n_agents)

        next_states = torch.FloatTensor(next_states).to(self.device)

        dones = torch.BoolTensor([[self.env.done] * self.env.n_agents] * self.batch_size)

        print("States Tensor Shape:", states.shape)
        print("Actions Tensor Shape:", actions.shape)
        print("Rewards Tensor Shape:", rewards.shape)
        print("Next States Tensor Shape:", next_states.shape)
        print("Dones Tensor Shape:", dones.shape)

        # Current Q-values
        state_action_values = self.network(states).gather(2, joint_actions.unsqueeze(-1)).squeeze(-1)

        # Expected Q-values
        next_state_values = torch.zeros(self.batch_size, self.env.n_agents).to(self.device)
        non_final_mask = ~dones.any(dim=1)
        non_final_next_states = next_states[non_final_mask]
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(2)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + rewards

        # Compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        print("Loss Tensor Shape:", loss.shape)

        self.optimizer.zero_grad()
        loss.backward()

        for name, param in self.network.named_parameters():
            if param.grad is not None:
                print(f"Gradient shape for {name}:", param.grad.shape)

        torch.nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def decay_epsilon(self):
        self.exploration_strategy.decay()


def game_loop(env, agent, training=True, num_episodes=10000, max_steps_per_episode=300, render=False, model_file='dqn_model'):
    total_rewards = []
    all_losses = []
    failure_count = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()

        print(f"Shape of obs: {np.shape(obs)}, Content: {obs}")
        obs_flat = flatten_obs(obs)
        print(f"Shape of flattened obs: {np.shape(obs_flat)}, Content: {obs_flat}")

        episode_reward = 0
        for step in range(max_steps_per_episode):
            if render:
                env.render()

            actions = agent.select_action(obs_flat)

            # print("Length of actions in game_loop:", len(actions))
            # print("Actions in game_loop:", actions)

            next_obs, reward, done, _ = env.step(actions)

            # print(f'---- Obs after step type: {type(next_obs)}')
            # print(next_obs)
            next_obs_flat = flatten_obs(next_obs)

            if training:
                agent.memory.push(obs_flat, actions, reward, next_obs_flat, done)
                loss = agent.train()
                all_losses.append(loss)

            episode_reward += reward
            obs = next_obs

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
            valid_losses = [loss for loss in all_losses[-100:] if loss is not None]
            if valid_losses:
                avg_loss = np.mean(valid_losses)
            else:
                avg_loss = "N/A" 

            if isinstance(avg_loss, str):
                print(f"Episode {episode}/{num_episodes}: Avg Reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}, Avg Loss: {avg_loss}, Epsilon: {agent.exploration_strategy.epsilon:.2f}")
            else:
                print(f"Episode {episode}/{num_episodes}: Avg Reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.exploration_strategy.epsilon:.2f}")


        # Save the model periodically
        if training and episode % 1000 == 0:
            torch.save(agent.network.state_dict(), model_file + f"_{episode}.pth")

    agent.exploration_strategy.decay()

    if training:
        torch.save(agent.network.state_dict(), model_file + "_final.pth")


if __name__ == "__main__":

    from macpp.core.environment import MACPPEnv
    from macpp.agents.exploration import EpsilonGreedy
    from macpp.core.environment import Action

    # Set up the environment
    env = MACPPEnv(
        grid_size=(3, 3), n_agents=2, n_pickers=1, n_objects=1, cell_size=300, debug_mode=False
    )

    # Set up exploration strategy
    epsilon_greedy_strategy = EpsilonGreedy(epsilon=1.0, min_epsilon=0.02, epsilon_decay=0.99)

    # Set up the agent
    agent = DQNAgent(env, epsilon_greedy_strategy,learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64, tau=0.1)

    # Train the agent
    game_loop(env, agent, training=True, num_episodes=2, max_steps_per_episode=200, render=False, model_file='dqn_model')

