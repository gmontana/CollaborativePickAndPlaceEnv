import numpy as np


class QTable:
    def __init__(self, n_agents, action_space):
        self.q_table = {}
        self.n_agents = n_agents
        self.action_space = action_space

    def initialise(self, state_hash):
        ''' 
        Initialising the Q table with small values may help exploration early on
        '''
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.random.uniform(-0.01, 0.01, (len(self.action_space),) * self.n_agents)
        return self.q_table[state_hash]

    def update(self, state, actions, value):
        state_key = self._state_to_key(state)
        current_q_values = self.initialise(state)
        action_indices = tuple(self.action_space.index(action) for action in actions)
        current_q_values[action_indices] = value

    def _state_to_key(self, state):
        if isinstance(state, list):
            return tuple(self._state_to_key(item) for item in state)
        elif isinstance(state, dict):
            return tuple((key, self._state_to_key(value)) for key, value in state.items())
        return state

    def count_elements(self):
        return sum([np.prod(v.shape) for v in self.q_table.values()])

    def get_q_values(self, state):
        current_q_values = self.initialise(state)
        return current_q_values

    def get_max_q_value(self, state_hash):
        current_q_values = self.initialise(state_hash)
        return np.max(current_q_values)

    def save_q_table(self, filename):
        print(f"Saving Q-value table: {filename}.")
        np.save(filename, self.q_table)
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def load_q_table(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()
        print(f"Number of elements in the Q table: {self.count_elements()}")


        print(f"Loading Q-value table: {filename}.")
class QLearning:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration=0.01,
        learning_rate_decay=0.995,
        min_learning_rate=0.01,
        max_steps_per_episode=50,
    ):
        self.env = env
        self.q_table = QTable(
            n_agents=env.n_agents, action_space=env.get_action_space()
        )
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.max_steps_per_episode = max_steps_per_episode

    def greedy_actions(self, state):
        q_values = self.q_table.initialise(state)
        actions_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
        return [self.q_table.action_space[index] for index in actions_indices]

    def epsilon_greedy_actions(self, state_hash):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return [
                np.random.choice(self.env.action_space)
                for _ in range(self.env.n_agents)  # Retrieve n_agents from the environment
            ]
        else:
            return self.greedy_actions(state_hash)

    def execute(self, state_hash):

        # Get actions, next state and rewards
        actions = self.greedy_actions(state_hash)
        next_state_hash, rewards, done = self.env.step(actions)
        total_reward = sum(rewards)
        return total_reward, done, next_state_hash
  
    def train(self, state_hash):

        # Get actions, next state and rewards
        actions = self.epsilon_greedy_actions(state_hash)
        next_state_hash, rewards, done = self.env.step(actions)
        total_reward = sum(rewards)

        # Update Q-values
        current_q_values = self.q_table.initialise(state_hash)
        max_next_q_value = self.q_table.get_max_q_value(next_state_hash)
        action_indices = tuple(self.q_table.action_space.index(action) for action in actions)
        if done:
            target = total_reward
        else:
            target = total_reward + self.discount_factor * max_next_q_value
        updated_value = current_q_values[action_indices] + self.learning_rate * (target - current_q_values[action_indices])
        self.q_table.update(state_hash, actions, updated_value)

        # Decay exploration and learning rates
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)

        return total_reward, done, next_state_hash
