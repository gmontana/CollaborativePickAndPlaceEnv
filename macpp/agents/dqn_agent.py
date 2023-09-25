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

    def select_action(self, state):
        sample = random.random()
        if sample < self.exploration_strategy.epsilon:
            # Exploration: Randomly select a joint action
            return [random.randrange(6) for _ in range(self.env.n_agents)]
        else:
            # Exploitation: Use the Q-network to select the best joint action
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                q_values = self.network(state)
                joint_action_idx = q_values.argmax().item()
                joint_action = self.decode_joint_action(joint_action_idx)
                return joint_action

    def encode_joint_action(self, joint_action):
        """
        Encode a joint action (list of individual actions) into a single joint action index.
        """
        joint_action_idx = 0
        for i, action in enumerate(joint_action):
            joint_action_idx += action * (6 ** i)
        return joint_action_idx

    def decode_joint_action(self, joint_action_idx):
        joint_action = []
        for _ in range(self.env.n_agents):
            action = joint_action_idx % 6
            joint_action.append(action)
            joint_action_idx //= 6
        return joint_action

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch from the replay buffer
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        print(f"states type and values: {type(states)} {states}")

        q_values = self.network(states)
        print(f"q_values type and values: {type(q_values)} {q_values}")

        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        print(f"actions type and shape: {type(actions)} {len(actions)}")

        # Expand the dimensions of the actions tensor
        actions_expanded = actions.unsqueeze(-1)
        print(f"action_expanded type and shape: {type(actions_expanded)} {len(actions_expanded)}")

        # Compute Q-values for the current states and actions
        state_action_values = q_values.gather(1, actions_expanded).squeeze(-1)

        # Compute the expected Q-values for the next states
        with torch.no_grad():
            non_final_mask = ~dones
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_network(next_states[non_final_mask]).max(1)[0].detach()

        # Compute the expected Q-values based on the Bellman equation
        expected_state_action_values = (rewards + (self.gamma * next_state_values)).unsqueeze(-1)

        # Compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


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
        obs_flat = flatten_obs(obs)
        episode_reward = 0
        for step in range(max_steps_per_episode):
            if render:
                env.render()
            actions = agent.select_action(obs_flat)
            # print("Actions in game_loop:", actions)
            next_obs, reward, done, _ = env.step(actions)
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
        if training:
            agent.decay_epsilon()
        if training and episode % 100 == 0:
            agent.update_target_network()
        if episode % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / 100
            success_rate = 1 - (failure_count / 100)
            valid_losses = [loss for loss in all_losses[-100:] if loss is not None]
            if valid_losses:
                avg_loss = np.mean(valid_losses)
            else:
                avg_loss = "N/A" 
            print(f"Episode {episode}/{num_episodes}: Avg Reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.exploration_strategy.epsilon:.2f}")
        if training and episode % 1000 == 0:
            torch.save(agent.network.state_dict(), model_file + f"_{episode}.pth")
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

