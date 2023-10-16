import wandb
# import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
# import torch.optim as optim
import gym
import random
import numpy as np
# import matplotlib.pyplot as plt
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')

SEED = 42
EPISODES = 2000
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION = 1.0
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
UPDATE_EVERY = 100
LOG_EVERY = 10
ALPHA = 0.4
#
FC1_DIMS = 1024
FC2_DIMS = 512

# DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs):
    flattened = []
    for _, agent_observations in obs.items():
        # Agent's own state
        flattened.extend(agent_observations['self']['position'])
        flattened.append(1 if agent_observations['self']['carrying_object'] is not None else 0)

        # Relative positions and distances to other agents
        for other_agent in agent_observations['agents']:
            dx = other_agent['position'][0] - agent_observations['self']['position'][0]
            dy = other_agent['position'][1] - agent_observations['self']['position'][1]
            flattened.extend([dx, dy])
            flattened.append(abs(dx) + abs(dy))

        # Relative positions and distances to objects
        for obj in agent_observations['objects']:
            dx = obj['position'][0] - agent_observations['self']['position'][0]
            dy = obj['position'][1] - agent_observations['self']['position'][1]
            flattened.extend([dx, dy])
            flattened.append(abs(dx) + abs(dy))

        # Relative positions and distances to goals
        for goal in agent_observations['goals']:
            dx = goal[0] - agent_observations['self']['position'][0]
            dy = goal[1] - agent_observations['self']['position'][1]
            flattened.extend([dx, dy])
            flattened.append(abs(dx) + abs(dy))

    return np.array(flattened)


class Network(nn.Module):
    def __init__(self, input_shape, num_actions, learning_rate, device):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape[0], 256)
        self.ln1 = nn.LayerNorm(256, elementwise_affine=False)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128, elementwise_affine=False)
        self.fc3 = nn.Linear(128, self.num_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def update_target(self, target_net):
        target_net.load_state_dict(self.state_dict())


class ReplayBuffer:

    def __init__(self, state_shape, capacity=MEM_SIZE, alpha=0.6, prioritized=False):

        self.capacity = capacity
        self.prioritized = prioritized
        self.mem_count = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.alpha = alpha

        if self.prioritized:
            self.priorities = np.ones(capacity)
            self.max_priority = 1.0

    def __len__(self):
        return min(self.mem_count, self.capacity)

    def add(self, state, action, reward, next_state, done):

        index = self.mem_count % self.capacity

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        if self.prioritized:
            self.priorities[index] = self.max_priority

        self.mem_count += 1

    def sample(self, batch_size, beta=0.4):

        if self.__len__() < batch_size:
            raise ValueError("Not enough experiences available for sampling.")

        # Using prioritised sampling
        if self.prioritized:

            probs = self.priorities[:self.__len__()]**self.alpha
            probs /= probs.sum()

            indices = np.random.choice(
                self.__len__(), batch_size, p=probs)

            weights = (len(probs) * probs[indices]) ** (-beta)
            weights /= weights.max()

        else:
            indices = np.random.choice(
                self.__len__(), batch_size)
            weights = np.ones(batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority ** self.alpha
                self.max_priority = max(self.max_priority, priority)


class DQNAgent:

    def __init__(self, env, state_shape):
        self.env = env

        self.exploration = EXPLORATION

        self.replay_buffer = ReplayBuffer(
            state_shape, capacity=MEM_SIZE, alpha=ALPHA, prioritized=True)

        self.policy_net = Network(
            state_shape, env.action_space_n, LEARNING_RATE, DEVICE)

        self.target_net = Network(
            state_shape, env.action_space_n, LEARNING_RATE, DEVICE)

        self.optimizer = optim.Adam(self.policy_net.parameters(
        ), lr=LEARNING_RATE)

    def get_action(self, state):
        if random.random() < self.exploration:  # Use instance variable, not constant
            return self.env.action_space.sample()
        else:
            return self.get_policy_action(state)

    def get_policy_action(self, state):
        q_values = self.policy_net(torch.from_numpy(state).float().to(DEVICE))
        return torch.argmax(q_values).item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_exploration(self, episode):
        self.exploration = max(
            EXPLORATION_MIN, EXPLORATION_MAX * (EXPLORATION_DECAY ** episode))

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def returning_epsilon(self):
        return self.exploration

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
            BATCH_SIZE, beta=0.4)

        # Convert to PyTorch tensors and send to device
        # states = torch.tensor(states, dtype=torch.float).to(DEVICE)
        states = torch.tensor([flatten_obs(s) for s in states], dtype=torch.float).to(
            DEVICE)

        actions = torch.tensor(actions, dtype=torch.long).to(
            DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float).to(DEVICE)
        # next_states = torch.tensor(next_states, dtype=torch.float).to(DEVICE)
        next_states = torch.tensor([flatten_obs(s) for s in next_states], dtype=torch.float).to(
            DEVICE)

        dones = torch.tensor(dones, dtype=torch.float).to(DEVICE)
        weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

        # Calculate Q-values and next Q-values
        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)

        # Calculate Q-targets
        q_targets = rewards + \
            (GAMMA * torch.max(next_q_values, dim=1)[0] * (1 - dones))

        # Extract the Q-values of the taken actions
        current_q_values = q_values.gather(
            1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute TD errors for PER
        td_errors = q_targets.detach() - current_q_values

        # Compute loss, considering possible priority weights
        loss = (F.mse_loss(current_q_values, q_targets,
                reduction='none') * weights).mean()

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(
            indices,
            td_errors.abs().cpu().detach().numpy()
        )

        return loss.item()


if __name__ == "__main__":

    from macpp.core.environment import MACPPEnv

    wandb.init(project='cpp', name='test_run', config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "architecture": "DQN",
        "fc1_dims": FC1_DIMS,
        "fc2_dims": FC2_DIMS,
        "epsilon_decay": EXPLORATION_DECAY,
    })

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MACPPEnv(grid_size=(3, 3), n_agents=2, n_pickers=1,
                   n_objects=1, cell_size=300, debug_mode=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observation_space = env.observation_space
    action_space = env.action_space_n

    sample_obs = env.reset()
    # print(sample_obs)
    flattened_obs_shape = flatten_obs(sample_obs).shape
    agent = DQNAgent(env, flattened_obs_shape)

    losses = []
    rewards = []
    average_reward = 0
    best_reward = 0

    for episode in tqdm(range(1, EPISODES + 1), desc="Training Progress"):
        state = env.reset()
        # state = np.array([state_tuple[0]])
        state_flat = flatten_obs(state)
        episode_return = 0
        loss_sum = 0
        num_updates = 0

        if episode % UPDATE_EVERY == 0:
            agent.update_target_net()

        while True:

            # Assuming adaptation in get_action()
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            # next_state = np.array([next_state_tuple])
            next_state_flat = flatten_obs(next_state)

            # Store transition in the replay buffer
            agent.replay_buffer.add(
                state_flat, action, reward, next_state_flat, done)

            # Train the agent
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)
                loss_sum += loss
            num_updates += 1
            state = next_state
            state_flat = next_state_flat
            episode_return += reward

            if done:
                if episode_return > best_reward:
                    best_reward = episode_return
                    agent.save_model("best_model.pth")
                average_reward += episode_return
                rewards.append(episode_return)
                break

        # Decay exploration rate after each episode
        agent.decay_exploration(episode)

        average_loss = loss_sum / num_updates if num_updates != 0 else 0
        average_reward = np.mean(rewards)

        # Log episodic metrics
        wandb.log({
            "episode_return": episode_return,
            "episode_avg_loss": average_loss,
            "episode_avg_reward": average_reward
        })

        if episode % LOG_EVERY == 0:
            tqdm.write(f"Episode: {episode}, "
                       f"Avg Reward: {np.mean(rewards[-LOG_EVERY:]):.2f}, "
                       f"Avg Loss: {average_loss:.2f}, "
                       f"Epsilon: {agent.returning_epsilon():.2f}")

    wandb.finish()
