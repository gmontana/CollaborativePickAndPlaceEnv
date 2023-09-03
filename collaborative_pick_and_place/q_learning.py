import numpy as np

class QTable:
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

    def save_q_table(self, filename):
        print(f"Saving Q-value table: {filename}.")
        np.save(filename, self.q_table)
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def load_q_table(self, filename):
        print(f"Loading Q-value table: {filename}.")
        self.q_table = np.load(filename, allow_pickle=True).item()
        print(f"Number of elements in the Q table: {self.count_elements()}")

    def get_best_action(self, state, env):
        hashed_state = self.hash_state(state)
        if hashed_state not in self.q_table:
            return random.choice(env.get_action_space())
        return np.argmax(self.q_table[hashed_state])


class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01, learning_rate_decay=0.995, min_learning_rate=0.01):
        self.env = env
        self.q_table = QTable(n_agents=env.n_agents, action_space=env.get_action_space())
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        # self.steps=0
        # self.final_return = 0

    def epsilon_greedy_actions(self, state_hash):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return [np.random.choice(self.q_table.action_space) for _ in range(self.q_table.n_agents)]
        else:
            return self.greedy_actions(state_hash)

    def greedy_actions(self, state_hash):
        q_values = self.q_table.initialise(state_hash)
        actions_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
        return [self.q_table.action_space[index] for index in actions_indices]

    def execute(self, max_num_steps):

        state = self.env.reset()
        state_hash = self.env.get_hashed_state()

        done = False
        success = False
        total_steps = 0
        total_return = 0

        while not done and total_steps < max_num_steps:  
            total_steps += 1

            # print(self.env.print_state())
            actions = self.greedy_actions(state_hash)
            # print(actions)
            _ , rewards, done = self.env.step(actions)
            next_state_hash = self.env.get_hashed_state()

            total_reward = sum(rewards)
            state_hash = next_state_hash
            total_return += total_reward

        if done: 
            success = True

        return success, total_return, total_steps


    def train(self, episodes, max_steps_per_episode):
        rewards_all_episodes = []
        successful_episodes = 0
        steps_per_episode = []

        for episode in range(episodes):
            state = self.env.reset()
            state_hash = self.env.get_hashed_state()
            steps = 0  # Local variable for steps in this episode
            rewards_current_episode = 0
            done = False

            for step in range(max_steps_per_episode):
                actions = self.epsilon_greedy_actions(state_hash)
                next_state, rewards, done = self.env.step(actions)
                next_state_hash = self.env.get_hashed_state()

                # the total reward is the sum of all agents' individual rewards
                total_reward = sum(rewards)

                current_q_values = self.q_table.initialise(state_hash)
                next_q_values = self.q_table.initialise(next_state_hash)
                action_indices = tuple(self.q_table.action_space.index(action) for action in actions)
                max_next_q_value = np.max(next_q_values)

                # Update Q-value using the Bellman equation
                if done:
                    target = total_reward
                else:
                    target = total_reward + self.discount_factor * max_next_q_value
                updated_value = current_q_values[action_indices] + self.learning_rate * (target - current_q_values[action_indices])
                self.q_table.update(state_hash, actions, updated_value)

                rewards_current_episode += total_reward
                state_hash = next_state_hash
                steps += 1

                if done:
                    break

            steps_per_episode.append(steps)
            rewards_all_episodes.append(rewards_current_episode)

            # Check for a successful episode
            if steps <= max_steps_per_episode:
                successful_episodes += 1

            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_all_episodes[-100:])
                avg_steps = np.mean(steps_per_episode[-100:])
                print(f"Episode {episode} | Average Reward: {avg_reward} | Average Steps: {avg_steps}")

        print('----------------------------------------------------------------')
        print(f"Number of successful episodes: {successful_episodes}/{episodes}")
        print(f"Average steps per episode: {sum(steps_per_episode) / episodes}")

        return self.q_table, rewards_all_episodes, steps_per_episode

