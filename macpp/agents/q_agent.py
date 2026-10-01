import os
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime

import numpy as np


class QTable:
    def __init__(self, initial_value=0.001, action_sizes=None):
        self.q_table = defaultdict(lambda: defaultdict(lambda: initial_value))
        self.initial_value = initial_value
        self.action_sizes = tuple(action_sizes) if action_sizes is not None else None

    def get_q_value(self, state_hash, actions):
        return self.q_table[state_hash][tuple(actions)]

    def set_q_value(self, state_hash, actions, value):
        self.q_table[state_hash][tuple(actions)] = value

    def get_max_q_value(self, state_hash):
        values = list(self.q_table[state_hash].values())
        if self.action_sizes is not None and len(values) < int(np.prod(self.action_sizes)):
            values.append(self.initial_value)
        return max(values, default=self.initial_value)

    def best_actions(self, state_hash):
        if self.action_sizes is not None:
            values = self.q_table.get(state_hash, {})
            best_value = self.get_max_q_value(state_hash)
            return random.choice(
                [
                    action
                    for action in np.ndindex(*self.action_sizes)
                    if values.get(action, self.initial_value) == best_value
                ]
            )
        if state_hash not in self.q_table:
            return None
        max_q_value = self.get_max_q_value(state_hash)
        best_acts = [
            act for act, q_value in self.q_table[state_hash].items() if q_value == max_q_value
        ]
        return random.choice(best_acts) if best_acts else None

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(
                {
                    "initial_value": self.initial_value,
                    "action_sizes": self.action_sizes,
                    "values": {state: dict(values) for state, values in self.q_table.items()},
                },
                file,
            )

    @classmethod
    def load(cls, filename, initial_value=0.0):
        with open(filename, "rb") as file:
            loaded_q_table = pickle.load(file)

        if "values" in loaded_q_table:
            initial_value = loaded_q_table["initial_value"]
            action_sizes = loaded_q_table["action_sizes"]
            loaded_q_table = loaded_q_table["values"]
        else:
            action_sizes = None
        instance = cls(initial_value, action_sizes)
        for state, values in loaded_q_table.items():
            instance.q_table[state].update(values)
        return instance


class ExplorationStrategy(ABC):
    @abstractmethod
    def select_action(self, agent, obs):
        pass


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, exploration_rate=1.0, min_exploration=0.01, exploration_decay=0.995):
        self.exploration_rate = exploration_rate
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay

    def select_action(self, agent, obs_hash):
        if random.random() < self.exploration_rate:
            return agent.env.action_space.sample().tolist()
        else:
            return agent.act(obs_hash, explore=False)

    def decay_exploration_rate(self):
        self.exploration_rate = max(
            self.min_exploration, self.exploration_rate * self.exploration_decay
        )


class UCB(ExplorationStrategy):
    def __init__(self, c=0.9):
        self.c = c

    def select_action(self, agent, obs):
        max_ucb = float("-inf")
        best_action = None

        # Generate all possible action combinations for the MultiDiscrete action space
        action_combinations = [list(x) for x in np.ndindex(*agent.env.action_space.nvec)]

        for action in action_combinations:
            q_value = agent.q_table.get_q_value(obs, action)
            state_count = max(1, agent.state_visits.get(obs, 0))
            state_action_key = (obs, tuple(action))
            state_action_count = agent.state_action_visits.get(state_action_key, 0)
            ucb_value = (
                float("inf")
                if state_action_count == 0
                else q_value + self.c * np.sqrt(np.log(state_count) / state_action_count)
            )

            if ucb_value > max_ucb:
                max_ucb = ucb_value
                best_action = action

        return best_action


class BaseAgent(ABC):
    def __init__(
        self,
        env,
        exploration_strategy,
        discount_factor,
        learning_rate,
        min_learning_rate,
        learning_rate_decay,
    ):
        self.env = env
        self.q_table = QTable(action_sizes=env.action_space.nvec)
        self.exploration_strategy = exploration_strategy
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.state_visits = defaultdict(int)
        self.state_action_visits = defaultdict(int)

    @abstractmethod
    def act(self, obs_hash, explore=False):
        pass

    @abstractmethod
    def learn(self, obs_hash, actions, next_state_hash, rewards, done):
        pass


class QLearning(BaseAgent):
    def __init__(
        self,
        env,
        exploration_strategy,
        discount_factor,
        learning_rate,
        min_learning_rate,
        learning_rate_decay,
    ):
        super().__init__(
            env,
            exploration_strategy,
            discount_factor,
            learning_rate,
            min_learning_rate,
            learning_rate_decay,
        )

    def act(self, obs_hash, explore=False):
        if explore:
            return self.exploration_strategy.select_action(self, obs_hash)
        else:
            best = self.q_table.best_actions(obs_hash)
            if best is None:
                return self.env.action_space.sample().tolist()
            return best

    def learn(self, obs_hash, actions, next_state_hash, rewards, done):
        max_next_q_value = self.q_table.get_max_q_value(next_state_hash)
        target = rewards + self.discount_factor * max_next_q_value if not done else rewards
        current_q_value = self.q_table.get_q_value(obs_hash, actions)
        updated_value = current_q_value + self.learning_rate * (target - current_q_value)
        self.q_table.set_q_value(obs_hash, actions, updated_value)


class DoubleQLearning(BaseAgent):
    def __init__(
        self,
        env,
        exploration_strategy,
        discount_factor,
        learning_rate,
        min_learning_rate,
        learning_rate_decay,
    ):
        super().__init__(
            env,
            exploration_strategy,
            discount_factor,
            learning_rate,
            min_learning_rate,
            learning_rate_decay,
        )
        self.q_table2 = QTable(action_sizes=env.action_space.nvec)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def act(self, obs_hash, explore=False):
        if explore:
            return self.exploration_strategy.select_action(self, obs_hash)
        else:
            actions = list(np.ndindex(*self.env.action_space.nvec))
            values = [
                self.q_table.get_q_value(obs_hash, action)
                + self.q_table2.get_q_value(obs_hash, action)
                for action in actions
            ]
            best_value = max(values)
            return random.choice(
                [action for action, value in zip(actions, values) if value == best_value]
            )

    def learn(self, obs_hash, actions, next_state_hash, rewards, done):
        selected, evaluated = (
            (self.q_table, self.q_table2)
            if random.random() < 0.5
            else (self.q_table2, self.q_table)
        )
        target = rewards
        if not done:
            next_action = selected.best_actions(next_state_hash)
            target += self.discount_factor * evaluated.get_q_value(next_state_hash, next_action)
        current = selected.get_q_value(obs_hash, actions)
        selected.set_q_value(obs_hash, actions, current + self.learning_rate * (target - current))


def game_loop(
    env,
    agent,
    training=False,
    num_episodes=1,
    msx_steps_per_episode=300,
    render=False,
    create_video=False,
    qtable_file=None,
):

    if msx_steps_per_episode <= 0:
        raise ValueError("The episode step limit must be positive.")

    total_steps = []
    total_returns = []
    total_success_rate = []
    total_avg_steps = []
    total_avg_returns = []
    total_failures = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs_hash = env.obs_to_hash(obs)
        done = False
        episode_steps = 0
        episode_returns = 0

        while not done:
            if training:
                actions = agent.act(obs_hash, explore=True)
            else:
                actions = agent.act(obs_hash, explore=False)

            # Update the state visit count
            agent.state_visits[obs_hash] = agent.state_visits.get(obs_hash, 0) + 1

            # Update state-action visit count
            state_action_key = (obs_hash, tuple(actions))
            agent.state_action_visits[state_action_key] = (
                agent.state_action_visits.get(state_action_key, 0) + 1
            )

            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            next_obs_hash = env.obs_to_hash(next_obs)

            episode_returns += rewards
            episode_steps += 1

            if training:
                agent.learn(obs_hash, actions, next_obs_hash, rewards, terminated)
            obs = next_obs
            obs_hash = next_obs_hash

            if render:
                env.render()
                time.sleep(0.1)

            if episode_steps >= msx_steps_per_episode or done:
                total_failures += int(not terminated)
                break

        total_steps.append(episode_steps)
        total_returns.append(episode_returns)

        avg_steps = np.mean(total_steps)
        avg_return = np.mean(total_returns)
        success_rate = (1 - (total_failures / (episode + 1))) * 100
        total_success_rate.append(success_rate)
        total_avg_steps.append(avg_steps)
        total_avg_returns.append(avg_return)

        if training:
            agent.learning_rate = max(
                agent.min_learning_rate, agent.learning_rate * agent.learning_rate_decay
            )
            strategy = agent.exploration_strategy
            decay = getattr(strategy, "decay_exploration_rate", getattr(strategy, "decay", None))
            if decay is not None:
                decay()

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}: Avg Steps: {avg_steps:.2f}, Avg Return: {avg_return:.2f}, Success rate: {success_rate:.0f}%, alpha: {agent.learning_rate:.3f}"
            )

    # create a video when needed
    if create_video:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = script_dir + f"/videos/episode_{timestamp}.mp4"
        print(f"Saving movie in {filename}.")
        env.save_video(filename)

    if training and qtable_file is not None:
        agent.q_table.save(qtable_file)
        if isinstance(agent, DoubleQLearning):
            agent.q_table2.save(str(qtable_file) + ".second")
    return {"steps": total_steps, "returns": total_returns, "success_rate": total_success_rate}


if __name__ == "__main__":
    from macpp.core.environment import MACPPEnv

    # Set up the environment
    env = MACPPEnv(
        grid_size=(3, 3), n_agents=2, n_pickers=1, n_objects=1, cell_size=300, debug_mode=True
    )

    # Set up exploration strategy
    epsilon_greedy_strategy = EpsilonGreedy(
        exploration_rate=1.0, min_exploration=0.02, exploration_decay=0.99
    )
    # ucb_strategy = UCB(c=5)

    # Set up the Q agent
    agent = DoubleQLearning(
        env,
        exploration_strategy=epsilon_greedy_strategy,
        # exploration_strategy=ucb_strategy,
        learning_rate=0.1,
        discount_factor=0.9,
        learning_rate_decay=0.995,
        min_learning_rate=0.01,
    )

    # Train the agent
    game_loop(
        env,
        agent,
        training=True,
        num_episodes=100,
        msx_steps_per_episode=300,
        render=False,
        qtable_file="qtable",
    )
