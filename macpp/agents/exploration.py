from abc import ABC, abstractmethod
import random
import numpy as np

class ExplorationStrategy(ABC):

    @abstractmethod
    def select_action(self, agent, obs):
        pass

class EpsilonGreedy(ExplorationStrategy):

    def __init__(self, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def select_action(self, agent, obs_hash):
        if random.random() < self.epsilon:
            return agent.env.action_space.sample().tolist()
        else:
            best = agent.q_table.best_actions(obs_hash)
            if best is None:
                return agent.env.action_space.sample().tolist()
            return best

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class UCB(ExplorationStrategy):

    def __init__(self, c=0.9):
        self.c = c

    def select_action(self, agent, obs):
        max_ucb = float('-inf')
        best_action = None

        # Generate all possible action combinations for the MultiDiscrete action space
        action_combinations = [list(x) for x in np.ndindex(*agent.env.action_space.nvec)]

        for action in action_combinations:
            q_value = agent.q_table.get_q_value(obs, action)
            state_count = agent.state_visits.get(obs, 1)  # Avoid division by zero
            state_action_key = (obs, tuple(action))
            state_action_count = agent.state_action_visits.get(state_action_key, 1)
            ucb_value = q_value + self.c * np.sqrt(np.log(state_count) / state_action_count)

            if ucb_value > max_ucb:
                max_ucb = ucb_value
                best_action = action

        return best_action
