"""Numerical regression tests for the optional learning examples."""

from types import SimpleNamespace

import numpy as np
import pytest

from macpp.agents.q_agent import DoubleQLearning, EpsilonGreedy, QLearning, QTable, game_loop
from macpp.core.environment import MACPPEnv


def test_qtable_save_load_and_unseen_actions(tmp_path):
    table = QTable(initial_value=0.25, action_sizes=(6, 6))
    table.set_q_value("state", [0, 0], -10)
    assert table.get_max_q_value("state") == 0.25
    assert table.best_actions("state") != (0, 0)
    filename = tmp_path / "qtable.pkl"
    table.save(filename)
    loaded = QTable.load(filename)
    assert loaded.get_q_value("state", [0, 0]) == -10
    assert loaded.get_q_value("state", [1, 1]) == 0.25
    assert loaded.get_q_value("new state", [1, 1]) == 0.25
    assert loaded.action_sizes == (6, 6)


def make_agent(kind=QLearning):
    return kind(
        MACPPEnv((3, 3), 2, 1, seed=1),
        EpsilonGreedy(),
        discount_factor=0.5,
        learning_rate=1,
        min_learning_rate=0.01,
        learning_rate_decay=0.9,
    )


def test_qlearning_terminal_target_does_not_bootstrap():
    agent = make_agent()
    agent.q_table.set_q_value("next", [0, 0], 100)
    agent.learn("state", [2, 3], "next", 7, True)
    assert agent.q_table.get_q_value("state", [2, 3]) == 7


@pytest.mark.parametrize("coin", [0.1, 0.9])
def test_double_q_selects_and_evaluates_with_different_tables(monkeypatch, coin):
    agent = make_agent(DoubleQLearning)
    monkeypatch.setattr("macpp.agents.q_agent.random.random", lambda: coin)
    selected, evaluated = (
        (agent.q_table, agent.q_table2) if coin < 0.5 else (agent.q_table2, agent.q_table)
    )
    selected.set_q_value("next", [0, 0], 20)
    selected.set_q_value("next", [1, 1], 10)
    evaluated.set_q_value("next", [0, 0], 4)
    evaluated.set_q_value("next", [1, 1], 100)
    agent.learn("state", [2, 3], "next", 3, False)
    assert selected.get_q_value("state", [2, 3]) == 5
    assert evaluated.get_q_value("state", [2, 3]) == evaluated.initial_value


def test_double_q_greedy_action_uses_both_estimates():
    agent = make_agent(DoubleQLearning)
    agent.q_table.set_q_value("state", [0, 0], 10)
    agent.q_table2.set_q_value("state", [1, 1], 20)
    assert agent.act("state") == (1, 1)


def test_training_loop_advances_state_and_respects_exact_step_limit():
    class Episode:
        obs_to_hash = staticmethod(MACPPEnv.obs_to_hash)

        def reset(self):
            self.index = 0
            return {"step": 0}, {}

        def step(self, actions):
            self.index += 1
            return {"step": self.index}, 1, self.index == 3, False, {}

    for training in (True, False):
        env = Episode()
        seen, learned = [], []
        agent = SimpleNamespace(
            act=lambda state, explore: seen.append(state) or [5, 5],
            learn=lambda state, actions, next_state, rewards, done: learned.append(
                (state, next_state, done)
            ),
            state_visits={},
            state_action_visits={},
            learning_rate=1,
            min_learning_rate=0.01,
            learning_rate_decay=0.5,
            exploration_strategy=EpsilonGreedy(),
        )
        metrics = game_loop(env, agent, training=training, msx_steps_per_episode=3)
        assert seen == [env.obs_to_hash({"step": i}) for i in range(3)]
        assert metrics == {"steps": [3], "returns": [3], "success_rate": [100]}
        assert agent.learning_rate == (0.5 if training else 1)
        if training:
            assert learned[-1][2] is True
            assert all(state != next_state for state, next_state, _ in learned)
        else:
            assert learned == []


def dqn_module():
    pytest.importorskip("torch")
    from macpp.agents import dqn_agent

    return dqn_agent


def test_dqn_all_joint_actions_decode_correctly(monkeypatch):
    module = dqn_module()
    torch = module.torch
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    agent = module.DQNAgent(MACPPEnv((3, 3), 2, 1), (2,))
    for parameter, target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(parameter, target)
        assert not target.requires_grad
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        for index in range(36):
            agent.policy_net.fc3.bias.zero_()
            agent.policy_net.fc3.bias[index] = 1
            action = agent.get_policy_action(np.zeros(2))
            assert action == list(divmod(index, 6))
            assert agent.env.action_space.contains(action)


def test_dqn_learning_gathers_joint_q_value_and_detaches_target(monkeypatch):
    module = dqn_module()
    torch = module.torch
    monkeypatch.setattr(module, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(module, "BATCH_SIZE", 2)
    agent = module.DQNAgent(MACPPEnv((3, 3), 2, 1), (2,))
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        agent.policy_net.fc3.bias[17] = 2  # [2, 5]
        agent.policy_net.fc3.bias[6] = 3  # [1, 0]
    for action, reward in [([2, 5], 7), ([1, 0], 11)]:
        agent.replay_buffer.add(np.zeros(2), action, reward, np.zeros(2), True)
    monkeypatch.setattr(
        agent.replay_buffer,
        "sample",
        lambda *args, **kwargs: (
            np.zeros((2, 2)),
            np.array([[2, 5], [1, 0]]),
            np.array([7, 11]),
            np.zeros((2, 2)),
            np.ones(2),
            np.arange(2),
            np.ones(2),
        ),
    )
    assert agent.learn() == pytest.approx(44.5)
    assert all(parameter.grad is None for parameter in agent.target_net.parameters())
    assert agent.replay_buffer.priorities[:2] == pytest.approx([5, 8])


def test_replay_priorities_apply_alpha_once_and_handle_zero_error(monkeypatch):
    module = dqn_module()
    replay = module.ReplayBuffer((1,), capacity=2, alpha=0.5)
    for _ in range(2):
        replay.add([0], [5, 5], 0, [0], False)
    replay.update_priorities([0, 1], [1, 16])
    captured = []

    def choose(size, batch_size, p):
        captured.append(p)
        return np.array([0, 1])

    monkeypatch.setattr(module.np.random, "choice", choose)
    replay.sample(2)
    assert captured[0] == pytest.approx([0.2, 0.8])
    replay.update_priorities([0, 1], [0, 0])
    result = replay.sample(2)
    assert np.isfinite(result[-1]).all()
    assert captured[1] == pytest.approx([0.5, 0.5])


def test_distance_features_encode_empty_hands():
    module = dqn_module()
    env = MACPPEnv((3, 3), 2, 1, seed=1)
    features = module.flatten_obs_dist(env.get_obs())
    assert features[3] == 0


def test_epsilon_greedy_uses_double_q_policy():
    agent = make_agent(DoubleQLearning)
    agent.exploration_strategy.exploration_rate = 0
    agent.q_table.set_q_value("state", [0, 0], 10)
    agent.q_table2.set_q_value("state", [1, 1], 20)
    assert agent.act("state", explore=True) == (1, 1)


def test_ucb_tries_unvisited_actions():
    from macpp.agents.exploration import UCB

    agent = make_agent()
    agent.state_visits["state"] = 1
    agent.state_action_visits[("state", (0, 0))] = 1
    assert UCB().select_action(agent, "state") == [0, 1]


def test_training_loop_with_real_environment():
    agent = make_agent()
    env = MACPPEnv((3, 3), 2, 1, seed=1)
    metrics = game_loop(env, agent, training=True, msx_steps_per_episode=3)
    assert metrics["steps"] == [3]
