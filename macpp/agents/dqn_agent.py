import wandb
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
import numpy as np
import torch.optim as optim
import platform
# import pdb
# import warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')

SEED = 1
EPISODES = 10000
LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 512
GAMMA = 0.998
EXPLORATION_MAX = 1.5
EXPLORATION_MIN = 0.095
EXPLORATION_DECAY = 0.9995
UPDATE_EVERY = 150  # how often to update the target network
ALPHA = 0.2

# Q network layer sizes
L1_DIM = 128
L2_DIM = 64

LOG_EVERY = 10  # how often to log the performance metrics
MAX_STEPS = 500  # max number of steps per episode

current_os = platform.system()
if current_os == "Darwin":  # macOS
    DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif current_os == "Linux":  # Linux
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs):
    """
    Flattens the observation dictionary into a 1D numpy array 

    Args:
    obs (dict): The observation dictionary.

    Returns:
    np.ndarray: The flattened observation array.
    """
    flattened_obs = []
    num_agents = len(obs)

    # Information for each agent
    for i in range(num_agents):
        agent_key = f'agent_{i}'
        agent_obs = obs[agent_key]

        # Self information
        flattened_obs.extend(agent_obs['self']['position'])
        flattened_obs.append(int(agent_obs['self']['picker']))
        flattened_obs.append(agent_obs['self']['carrying_object']
                             if agent_obs['self']['carrying_object'] is not None else -1)

        # Other agents' information
        for other_agent in agent_obs['agents']:
            flattened_obs.extend(other_agent['position'])
            flattened_obs.append(int(other_agent['picker']))
            flattened_obs.append(
                other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

    # Objects' information
    for obj in obs['agent_0']['objects']:
        flattened_obs.extend(obj['position'])

    # Goals' information
    for goal in obs['agent_0']['goals']:
        flattened_obs.extend(goal)

    return np.array(flattened_obs)


def flatten_obs_dist(obs):
    """
    Convert the structured observation of the environment into a flat numpy array, capturing the relative positions,
    Manhattan distances, and picker status of agents, objects, and goals from the perspective of each agent.

    The function iterates through each agent's perspective in the observation dictionary. For each agent, it calculates
    the relative positions (dx, dy) and the Manhattan distances of other agents, objects, and goals with respect to the
    agent's own position. The agent's own position, picker status, and a binary indicator of whether it is carrying an
    object are also included in the output array.

    The output array follows a specific pattern for each agent:
    - Agent's own position (x, y)
    - Binary indicator: 1 if the agent is a picker, 0 otherwise
    - Binary indicator: 1 if carrying an object, 0 otherwise
    - For each other agent:
        - Relative position (dx, dy)
        - Manhattan distance (|dx| + |dy|)
        - Binary indicator: 1 if the other agent is a picker, 0 otherwise
    - For each object:
        - Relative position (dx, dy)
        - Manhattan distance (|dx| + |dy|)
    - For each goal:
        - Relative position (dx, dy)
        - Manhattan distance (|dx| + |dy|)

    Args:
    obs (dict): The observation of the environment, structured as a dictionary with agent IDs as keys and dictionaries
                of their respective observations as values.

    Returns:
    np.ndarray: A flat numpy array representing the relative positions, Manhattan distances, and picker status of
                entities in the environment from the perspective of each agent.
    """
    flattened_obs = []
    for agent_id, agent_obs in obs.items():
        # Agent's own position and picker status
        flattened_obs.extend(agent_obs['self']['position'])
        flattened_obs.append(int(agent_obs['self']['picker']))
        flattened_obs.append(
            1 if agent_obs['self']['carrying_object'] is not None else 0)

        # Other agents
        for other_agent in agent_obs['agents']:
            dx = other_agent['position'][0] - agent_obs['self']['position'][0]
            dy = other_agent['position'][1] - agent_obs['self']['position'][1]
            manhattan_distance = abs(dx) + abs(dy)
            flattened_obs.extend(
                [dx, dy, manhattan_distance, int(other_agent['picker'])])

        # Objects
        for obj in agent_obs['objects']:
            dx = obj['position'][0] - agent_obs['self']['position'][0]
            dy = obj['position'][1] - agent_obs['self']['position'][1]
            manhattan_distance = abs(dx) + abs(dy)
            flattened_obs.extend([dx, dy, manhattan_distance])

        # Goals
        for goal in agent_obs['goals']:
            dx = goal[0] - agent_obs['self']['position'][0]
            dy = goal[1] - agent_obs['self']['position'][1]
            manhattan_distance = abs(dx) + abs(dy)
            flattened_obs.extend([dx, dy, manhattan_distance])

    return np.array(flattened_obs)


def obs_to_grid(obs, grid_size):
    '''
    Converts the observation dictionary into a 3D grid representation.

    The grid is represented as a 3D NumPy array with dimensions (grid_width, grid_length, 3),
    where the last dimension corresponds to different channels for agents, objects, and goals.
    Each cell in the grid can be either 0 or 1, indicating the absence or presence of an entity.

    Args:
    obs (dict): The observation dictionary containing information about agents, objects, and goals.
    grid_size (tuple): A tuple representing the size of the grid as (grid_width, grid_length).

    Returns:
    np.ndarray: A 3D NumPy array representing the grid.
    '''
    grid_width, grid_length = grid_size
    grid = np.zeros((grid_width, grid_length, 3))

    for _, agent_data in obs.items():
        x, y = agent_data['self']['position']
        grid[x, y, 0] = 1

        for other_agent in agent_data['agents']:
            x, y = other_agent['position']
            grid[x, y, 0] = 1

        for obj in agent_data['objects']:
            x, y = obj['position']
            grid[x, y, 1] = 1

        for goal in agent_data['goals']:
            x, y = goal
            grid[x, y, 2] = 1

    return grid


class Network(nn.Module):
    def __init__(self, input_shape, num_actions, learning_rate, device):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape[0], L1_DIM)
        self.ln1 = nn.LayerNorm(L1_DIM, elementwise_affine=False)
        self.fc2 = nn.Linear(L1_DIM, L2_DIM)
        self.ln2 = nn.LayerNorm(L2_DIM, elementwise_affine=False)
        self.fc3 = nn.Linear(L2_DIM, self.num_actions)

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

    def __init__(self, state_shape, capacity=MEM_SIZE, alpha=0.6, prioritized=True):

        self.capacity = capacity
        self.prioritized = prioritized
        self.mem_count = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=object)
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
        self.actions[index] = list(action)
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        if self.prioritized:
            self.priorities[index] = self.max_priority

        self.mem_count += 1

    def sample(self, batch_size, beta=0.4):

        if self.__len__() < batch_size:
            raise ValueError("Not enough experiences available for sampling.")

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

        self.device = DEVICE

        self.env = env

        self.exploration = EXPLORATION_MAX

        self.replay_buffer = ReplayBuffer(
            state_shape, capacity=MEM_SIZE, alpha=ALPHA, prioritized=True)

        self.policy_net = Network(
            state_shape, env.action_space_n, LEARNING_RATE, self.device)

        self.target_net = Network(
            state_shape, env.action_space_n, LEARNING_RATE, self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(
        ), lr=LEARNING_RATE)

    def get_action(self, state):
        if random.random() < self.exploration:
            return self.env.action_space.sample()
        else:
            return self.get_policy_action(state)

    def get_policy_action(self, state):
        q_values = self.policy_net(
            torch.from_numpy(state).float().to(self.device))
        actions = torch.argmax(q_values.view(len(self.env.agents), -1), dim=1)
        return actions.tolist()

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

        # sample from the buffer
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(
            BATCH_SIZE, beta=0.4)

        # convert the states to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float).to(self.device)

        # reshape and convert actions to tensors
        actions = torch.tensor(
            np.stack(actions), dtype=torch.long).to(self.device)

        # convert other quantities
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.device)

        # Calculate Q-values and next Q-values
        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)

        # Calculate Q-targets
        q_targets = rewards + \
            (GAMMA * torch.max(next_q_values, dim=1)[0] * (1 - dones))

        # Extract the Q-values of the taken actions
        num_agents = actions.size(1)
        q_values_selected = torch.zeros(actions.size(0)).to(self.device)
        for i in range(num_agents):
            q_values_selected += q_values[torch.arange(
                actions.size(0)), actions[:, i]]

        # Compute TD errors for PER
        td_errors = q_targets.detach() - q_values_selected

        # Compute loss, considering possible priority weights
        loss = (F.mse_loss(q_values_selected, q_targets,
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

    wandb.init(project='cpp_dqn', name='test_run', config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "architecture": "DQN",
        "epsilon_decay": EXPLORATION_DECAY,
    })

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MACPPEnv(grid_size=(3, 3), n_agents=2, n_pickers=1,
                   n_objects=1, cell_size=300, debug_mode=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observation_space = env.observation_space
    action_space = env.action_space_n
    grid_width, grid_length = env.grid_width, env.grid_length

    sample_obs, _ = env.reset()
    flattened_obs = flatten_obs(sample_obs)
    # grid_obs = obs_to_grid(sample_obs, (grid_width, grid_length))
    # print(f"Flattened obs: {flattened_obs}")
    # print(f"Sample obs: {sample_obs}")
    # print(f"Grid obs: {grid_obs}")

    agent = DQNAgent(env, (len(flattened_obs),))

    losses = []
    rewards = []
    rewards_deque = deque(maxlen=LOG_EVERY)
    total_steps = 0
    failed_episodes = 0

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        state_flat = flatten_obs(state)
        episode_reward = 0
        loss_sum = 0
        num_updates = 0
        episode_steps = 0

        if episode % UPDATE_EVERY == 0:
            agent.update_target_net()

        for t in range(MAX_STEPS):

            action = agent.get_action(state_flat)
            next_state, reward, done, _ = env.step(action)
            next_state_flat = flatten_obs(next_state)
            # print(f"Action: {action}")
            # print(f"Next state: {next_state}")

            agent.replay_buffer.add(
                state_flat, action, reward, next_state_flat, done)

            # pdb.set_trace()

            loss = agent.learn()

            if loss is not None:
                losses.append(loss)
                loss_sum += loss
            num_updates += 1

            state = next_state
            state_flat = next_state_flat

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if done:
                break

        if episode_steps == MAX_STEPS:
            failed_episodes += 1

        agent.decay_exploration(episode)

        rewards.append(episode_reward)

        if episode % LOG_EVERY == 0:
            recent_rewards = rewards[-LOG_EVERY:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else np.nan
            avg_loss = loss_sum / num_updates if num_updates != 0 else np.nan
            failure_rate = failed_episodes / LOG_EVERY
            avg_steps = total_steps / LOG_EVERY

            print(f"Episode: {episode:06}, "
                  f"Avg Reward: {avg_reward:06.2f}, "
                  f"Avg Loss: {avg_loss:06.2f}, "
                  f"Epsilon: {agent.returning_epsilon():03.2f}, "
                  f"Failure Rate: {failure_rate:03.2f}, "
                  f"Avg Steps: {avg_steps:03.2f}")

            wandb.log({"Average Reward": avg_reward,
                       "Average Loss": avg_loss,
                       "Epsilon": agent.returning_epsilon(),
                       "Failure Rate": failure_rate,
                       "Average Steps": avg_steps})

            failed_episodes = 0
            total_steps = 0

    wandb.finish()
