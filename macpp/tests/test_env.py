"""Behavioral checks for game rules, Gymnasium integration and rendering."""

import random
from copy import deepcopy
from itertools import product

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import macpp  # noqa: F401 -- registers the environments
from macpp.core.environment import MACPPEnv


def make_env(agents=None, objects=None, goals=None, **kwargs):
    agents = agents or [((0, 0), True, None), ((4, 4), False, None)]
    objects = objects or [(2, 2)]
    goals = goals or [(3, 3)]
    state = {
        "agents": {
            f"agent_{i}": {"position": pos, "picker": picker, "carrying_object": obj}
            for i, (pos, picker, obj) in enumerate(agents)
        },
        "objects": [{"id": i, "position": pos} for i, pos in enumerate(objects)],
        "goals": goals,
    }
    return MACPPEnv(
        (5, 5), len(agents), sum(a[1] for a in agents), len(objects), initial_state=state, **kwargs
    )


def assert_invariants(env):
    assert len({agent.position for agent in env.agents}) == env.n_agents
    assert len({obj.position for obj in env.objects}) == env.n_objects
    assert sorted(obj.id for obj in env.objects) == list(range(env.n_objects))
    assert sum(agent.picker for agent in env.agents) == env.n_pickers
    for obj in env.objects:
        carriers = [agent for agent in env.agents if agent.carrying_object is obj]
        assert len(carriers) <= 1
        assert obj.carrying_agent is (carriers[0] if carriers else None)
        if carriers:
            assert obj.position == carriers[0].position
    assert env.observation_space.contains(env.get_obs())


@pytest.mark.parametrize(
    "action,expected",
    [(0, (0, 0)), (1, (0, 1)), (2, (0, 0)), (3, (1, 0)), (4, (0, 0)), (5, (0, 0))],
)
def test_movement_and_step_reward(action, expected):
    env = make_env()
    _, reward, terminated, truncated, _ = env.step([action, 5])
    assert env.agents[0].position == expected
    assert reward == -2
    assert not terminated and not truncated


def test_rectangular_grid_boundaries():
    env = MACPPEnv((7, 2), 2, 1, seed=5)
    env.agents[0].position = (6, 1)
    env.agents[1].position = (0, 0)
    env.step([3, 0])
    env.step([1, 2])
    assert [a.position for a in env.agents] == [(6, 1), (0, 0)]


@pytest.mark.parametrize(
    "positions,actions,expected",
    [
        ([(0, 0), (1, 0)], [3, 2], [(0, 0), (1, 0)]),  # swaps blocked
        ([(0, 0), (2, 0)], [3, 2], [(1, 0), (2, 0)]),  # lower index wins
        ([(1, 0), (0, 0)], [3, 3], [(2, 0), (1, 0)]),  # follow a vacated cell
        ([(0, 0), (1, 0)], [3, 3], [(0, 0), (2, 0)]),  # occupied when processed
    ],
)
def test_sequential_collisions(positions, actions, expected):
    env = make_env(agents=[(positions[0], True, None), (positions[1], False, None)])
    env.step(actions)
    assert [a.position for a in env.agents] == expected


def test_pickup_pass_drop_complete_episode():
    env = make_env(
        agents=[((0, 0), True, None), ((2, 0), False, None)], objects=[(1, 0)], goals=[(3, 0)]
    )
    assert env.step([3, 5])[1:4] == (8, False, False)
    assert env.agents[0].carrying_object is env.objects[0]
    assert env.step([4, 4])[1:4] == (8, False, False)
    assert env.agents[1].carrying_object is env.objects[0]
    assert env.objects[0].position == (2, 0)
    assert env.step([5, 3])[1:4] == (108, True, False)
    assert env.objects[0].position == (3, 0)
    assert env.objects[0].carrying_agent is None
    assert_invariants(env)
    with pytest.raises(RuntimeError, match="reset"):
        env.step([5, 5])
    env.reset()
    assert not env.done
    assert env.objects[0].position == (1, 0)


def test_picker_cannot_drop_and_dropper_cannot_pick():
    env = make_env(
        agents=[((0, 0), True, 0), ((2, 0), False, None)],
        objects=[(0, 0), (2, 0)],
        goals=[(1, 0), (3, 0)],
    )
    assert env.step([3, 5])[1:4] == (-2, False, False)
    assert env.agents[0].carrying_object is env.objects[0]
    assert env.agents[1].carrying_object is None


def test_automatic_pickup_on_wait_and_pass():
    for action in (4, 5):
        env = make_env(agents=[((0, 0), True, None), ((1, 0), False, None)], objects=[(0, 0)])
        _, reward, _, _, _ = env.step([action, action])
        assert reward == (18 if action == 4 else 8)
        assert env.agents[1 if action == 4 else 0].carrying_object is env.objects[0]


def test_pass_onto_goal_drops_in_same_step():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, None)], objects=[(0, 0)], goals=[(1, 0)]
    )
    assert env.step([4, 4])[1:4] == (118, True, False)
    assert env.objects[0].carrying_agent is None


def test_delivered_boxes_remain_delivered_and_cannot_farm_rewards():
    env = make_env(
        agents=[((0, 0), True, None), ((4, 4), False, None)],
        objects=[(1, 0), (2, 2)],
        goals=[(1, 0), (3, 3)],
    )
    assert env.step([3, 5])[1] == -2
    assert env.agents[0].carrying_object is None
    assert env.objects[0].position == (1, 0)
    assert env.step([5, 5])[1] == -2


def test_carried_boxes_block_other_carried_boxes():
    env = make_env(
        agents=[((0, 0), True, 0), ((4, 4), False, None)],
        objects=[(0, 0), (1, 0)],
        goals=[(1, 0), (3, 3)],
    )
    env.step([3, 5])
    assert env.agents[0].position == (0, 0)
    assert_invariants(env)


@pytest.mark.parametrize(
    "giver_picker,receiver_picker,expected",
    [
        (True, False, 6),
        (False, True, -14),
        (True, True, -4),
        (False, False, -4),
    ],
)
def test_pass_rewards(giver_picker, receiver_picker, expected):
    agents = [
        ((0, 0), giver_picker, 0),
        ((1, 0), receiver_picker, None),
        ((4, 4), True, None),
        ((4, 3), False, None),
    ]
    env = make_env(agents=agents, objects=[(0, 0)])
    assert env.step([4, 4, 5, 5])[1] == expected
    assert env.agents[1].carrying_object is env.objects[0]
    assert_invariants(env)


@pytest.mark.parametrize(
    "receiver,actions,blocked_box",
    [
        ((1, 0), [4, 5], False),
        ((1, 0), [5, 4], False),
        ((2, 0), [4, 4], False),
        ((1, 1), [4, 4], False),
        ((1, 0), [4, 4], True),
    ],
)
def test_invalid_passes_do_not_transfer(receiver, actions, blocked_box):
    objects = [(0, 0), receiver] if blocked_box else [(0, 0)]
    goals = [(3, 3), (3, 4)] if blocked_box else [(3, 3)]
    env = make_env(
        agents=[((0, 0), True, 0), (receiver, False, None)], objects=objects, goals=goals
    )
    assert env.step(actions)[1] == -2
    assert env.agents[0].carrying_object is env.objects[0]


def test_two_carriers_cannot_swap_boxes():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, 1)],
        objects=[(0, 0), (1, 0)],
        goals=[(3, 3), (3, 4)],
    )
    assert env.step([4, 4])[1] == -2
    assert [a.carrying_object.id for a in env.agents] == [0, 1]


def test_good_pass_priority_before_reserving_receivers():
    env = make_env(
        agents=[((0, 0), False, 0), ((1, 0), False, None), ((2, 0), True, 1)],
        objects=[(0, 0), (2, 0)],
        goals=[(3, 3), (3, 4)],
    )
    assert env.step([4, 4, 4])[1] == 7
    assert env.agents[1].carrying_object.id == 1
    assert env.agents[0].carrying_object.id == 0


def test_giver_does_not_reserve_multiple_receivers():
    env = make_env(
        agents=[((1, 1), True, 0), ((2, 1), False, None), ((1, 2), False, None), ((1, 3), True, 1)],
        objects=[(1, 1), (1, 3)],
        goals=[(3, 3), (3, 4)],
    )
    assert env.step([4] * 4)[1] == 16
    assert [env.agents[i].carrying_object.id for i in (1, 2)] == [0, 1]
    assert_invariants(env)


def test_contested_receiver_ties_follow_agent_order():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, None), ((2, 0), True, 1)],
        objects=[(0, 0), (2, 0)],
        goals=[(3, 3), (3, 4)],
    )
    env.step([4, 4, 4])
    assert env.agents[1].carrying_object.id == 0
    assert_invariants(env)


def test_box_cannot_travel_down_pass_chain_in_one_step():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, None), ((2, 0), False, None)], objects=[(0, 0)]
    )
    env.step([4, 4, 4])
    assert env.agents[1].carrying_object is env.objects[0]
    assert env.agents[2].carrying_object is None


@pytest.mark.parametrize(
    "actions",
    [
        None,
        [],
        [5],
        [5, 5, 5],
        [0.5, 5],
        [True, 5],
        [6, 5],
        [-1, 5],
        [None, 5],
        ["1", 5],
        [[1], [5]],
        np.array(5),
    ],
)
def test_invalid_actions_rejected_before_mutation(actions):
    env = make_env()
    before = env.get_obs()
    with pytest.raises(ValueError):
        env.step(actions)
    assert env.get_obs() == before
    assert env._get_rewards() == [0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size": (0, 5)},
        {"grid_size": (2.5, 5)},
        {"grid_size": (5,)},
        {"n_agents": 1},
        {"n_agents": True},
        {"n_pickers": 0},
        {"n_pickers": 2},
        {"n_objects": 0},
        {"n_objects": -1},
        {"n_objects": 2.5},
        {"cell_size": None},
        {"cell_size": 0},
        {"render_mode": "unknown"},
        {"grid_size": (3, 3), "n_objects": 4},
    ],
)
def test_invalid_configuration(kwargs):
    config = dict(grid_size=(5, 5), n_agents=2, n_pickers=1)
    config.update(kwargs)
    with pytest.raises(ValueError):
        MACPPEnv(**config)


def test_seeded_reset_and_constructor_are_reproducible_and_local():
    a = MACPPEnv((5, 5), 4, 2, 3, seed=123)
    b = MACPPEnv((5, 5), 4, 2, 3, seed=123)
    assert a.get_obs() == b.get_obs()
    for _ in range(5):
        random.random()
        np.random.random()
        assert a.reset()[0] == b.reset()[0]
    first, _ = a.reset(seed=9)
    assert a.reset(seed=9)[0] == first
    cells = [agent.position for agent in a.agents] + [obj.position for obj in a.objects] + a.goals
    assert len(set(cells)) == len(cells)


def test_observations_and_initial_state_cannot_mutate_live_state():
    env = make_env()
    obs, _ = env.reset()
    obs["agent_0"]["self"]["position"] = (4, 4)
    obs["agent_0"]["objects"][0]["position"] = (0, 0)
    obs["agent_0"]["goals"] += ((0, 0),)
    assert env.agents[0].position == (0, 0)
    assert env.objects[0].position == (2, 2)
    assert env.goals == [(3, 3)]
    original = deepcopy(env.initial_state)
    other = MACPPEnv((5, 5), 2, 1, initial_state=original)
    original["goals"][0] = (0, 0)
    assert other.reset()[0]["agent_0"]["goals"] == ((3, 3),)


def test_initial_state_order_does_not_reassign_agent_identity():
    env = make_env()
    state = deepcopy(env.initial_state)
    state["agents"] = dict(reversed(list(state["agents"].items())))
    env.reset_from_obs(state)
    assert env.agents[0].position == (0, 0)
    assert env.agents[0].picker


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["agents"].pop("agent_1"),
        lambda s: s["agents"]["agent_0"].update(position=(5, 0)),
        lambda s: s["agents"]["agent_1"].update(position=(0, 0)),
        lambda s: s["agents"]["agent_0"].update(picker=False),
        lambda s: s["agents"]["agent_0"].update(carrying_object=7),
        lambda s: s["agents"]["agent_0"].update(carrying_object=0),  # wrong position
        lambda s: s["objects"][0].update(id=1),
        lambda s: s["objects"].clear(),
        lambda s: s["goals"].clear(),
        lambda s: s["goals"].append((3, 3)),
        lambda s: s["goals"].__setitem__(0, (-1, 0)),
    ],
)
def test_invalid_initial_states_are_rejected_atomically(mutation):
    env = make_env()
    before = env.get_obs()
    state = deepcopy(env.initial_state)
    mutation(state)
    with pytest.raises(ValueError):
        env.reset_from_obs(state)
    assert env.get_obs() == before


def test_duplicate_ownership_objects_and_goals_are_rejected():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, 1)],
        objects=[(0, 0), (1, 0)],
        goals=[(3, 3), (3, 4)],
    )
    for field in ("ownership", "objects", "goals"):
        state = deepcopy(env.initial_state)
        if field == "ownership":
            state["agents"]["agent_1"].update(position=(0, 0), carrying_object=0)
        elif field == "objects":
            state["objects"][1]["id"] = 0
        else:
            state["goals"][1] = state["goals"][0]
        with pytest.raises(ValueError):
            env.reset_from_obs(state)


def test_gymnasium_checker():
    check_env(MACPPEnv((5, 5), 4, 2, 3), skip_render_check=True)


ENV_IDS = sorted(key for key in gym.envs.registry if key.startswith("macpp-"))


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_registered_environments_and_random_rollout_invariants(env_id):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=9)
    env.action_space.seed(42)
    assert env.observation_space.contains(obs)
    delivered_before = set()
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert_invariants(env.unwrapped)
        delivered = {
            obj.id
            for obj in env.unwrapped.objects
            if obj.carrying_agent is None and obj.position in env.unwrapped.goals
        }
        assert delivered_before <= delivered
        delivered_before = delivered
        if terminated or truncated:
            break
    env.close()


def test_every_two_agent_action_combination_preserves_invariants():
    env = make_env(
        agents=[((0, 0), True, 0), ((1, 0), False, None)], objects=[(0, 0)], goals=[(2, 0)]
    )
    for actions in product(range(6), repeat=2):
        env.reset()
        env.step(actions)
        assert_invariants(env)


def test_rgb_render_and_recording(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = make_env(cell_size=20, render_mode="rgb_array", create_video=True)
    first = env.render()
    assert first.shape == (100, 100, 3)
    assert first.dtype == np.uint8
    env.step([3, 5])
    assert len(env.frames) == 1
    assert np.array_equal(env.frames[0], env.render())
    assert not np.array_equal(first, env.frames[0])
    env.close()
    assert env.render().shape == first.shape
    env.close()
    env.reset()
    assert env.frames == []
    with pytest.raises(ValueError, match="No frames"):
        env.save_video(tmp_path / "empty.mp4")


def test_goal_color_tracks_delivery_even_when_picker_stands_on_it():
    env = make_env(
        agents=[((1, 0), True, None), ((4, 4), False, None)],
        objects=[(1, 0), (2, 2)],
        goals=[(1, 0), (3, 3)],
        cell_size=20,
        render_mode="rgb_array",
    )
    assert tuple(env.render()[0, 30]) == (0, 255, 0)
    env.close()


def test_video_export(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    env = make_env(create_video=True, cell_size=20)
    env.step([3, 5])
    env.step([1, 5])
    filename = tmp_path / "video" / "episode.mp4"
    env.save_video(filename)
    assert filename.stat().st_size > 0
    env.close()
