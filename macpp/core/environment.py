"""Cooperative grid-world with sequential movement and coordinated box transfers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REWARD_STEP = -1
REWARD_GOOD_PASS = 5
REWARD_BAD_PASS = -5
REWARD_DROP = 10
REWARD_PICKUP = 10
REWARD_COMPLETION = 50


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PASS = 4
    WAIT = 5

    @staticmethod
    def is_valid(action):
        return (
            isinstance(action, Integral)
            and not isinstance(action, (bool, np.bool_))
            and action in Action._value2member_map_
        )


class Object:
    def __init__(self, position: tuple[int, int], id: int):
        self._position = position
        self.id = id
        self.carrying_agent: Optional[Agent] = None

    @property
    def position(self):
        return self.carrying_agent.position if self.carrying_agent else self._position

    @position.setter
    def position(self, value):
        self._position = value

    def get_object_obs(self):
        return {"id": self.id, "position": self.position}


class Agent:
    def __init__(self, position, picker, carrying_object=None, reward=0):
        self.position = position
        self.picker = picker
        self.carrying_object = carrying_object
        self.reward = reward
        if carrying_object is not None:
            if carrying_object.carrying_agent is not None:
                raise ValueError("An object cannot be carried by two agents.")
            carrying_object.carrying_agent = self

    def move_up(self):
        x, y = self.position
        self.position = (x, max(0, y - 1))

    def move_down(self, grid_length):
        x, y = self.position
        self.position = (x, min(grid_length - 1, y + 1))

    def move_left(self):
        x, y = self.position
        self.position = (max(0, x - 1), y)

    def move_right(self, grid_width):
        x, y = self.position
        self.position = (min(grid_width - 1, x + 1), y)

    def pick_up(self, obj):
        if (
            self.picker
            and self.carrying_object is None
            and obj.carrying_agent is None
            and obj.position == self.position
        ):
            self.carrying_object = obj
            obj.carrying_agent = self
            self.reward += REWARD_PICKUP

    def drop(self, obj):
        if not self.picker and self.carrying_object is obj:
            obj.position = self.position
            obj.carrying_agent = None
            self.carrying_object = None
            self.reward += REWARD_DROP

    def pass_object(self, other_agent):
        if (
            self.carrying_object is not None
            and other_agent.carrying_object is None
            and sum(abs(a - b) for a, b in zip(self.position, other_agent.position)) == 1
        ):
            other_agent.carrying_object = self.carrying_object
            self.carrying_object.carrying_agent = other_agent
            self.carrying_object = None

    def get_basic_agent_obs(self):
        return {
            "position": self.position,
            "picker": int(self.picker),
            "carrying_object": self.carrying_object.id if self.carrying_object else -1,
        }

    def get_agent_obs(self, all_agents, all_objects, goals):
        return {
            "self": self.get_basic_agent_obs(),
            "agents": tuple(
                agent.get_basic_agent_obs() for agent in all_agents if agent is not self
            ),
            "objects": tuple(obj.get_object_obs() for obj in all_objects),
            "goals": tuple(goals),
        }


class MACPPEnv(gym.Env):
    """All agents share a team reward and a fully observed world.

    Moves resolve in agent-index order. Pickups and drops are automatic, while
    a transfer requires both adjacent agents to choose PASS. Delivered boxes
    stay on their goals. See README.md for the complete transition order.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: tuple[int, int],
        n_agents: int,
        n_pickers: int,
        n_objects: Optional[int] = 1,
        initial_state=None,
        cell_size: int = 100,
        debug_mode: bool = False,
        create_video: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
            raise ValueError("grid_size must contain (width, height).")
        self.grid_width, self.grid_length = (
            self._positive_int(value, "grid dimension") for value in grid_size
        )
        self.n_agents = self._positive_int(n_agents, "n_agents")
        self.n_pickers = self._positive_int(n_pickers, "n_pickers")
        self.n_objects = self._positive_int(
            n_agents if n_objects is None else n_objects, "n_objects"
        )
        if not 0 < self.n_pickers < self.n_agents:
            raise ValueError("At least one picker and one dropper are required.")
        self.cell_size = self._positive_int(cell_size, "cell_size")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")
        # Random layouts allocate a distinct cell to every agent, object and goal.
        if (
            initial_state is None
            and self.n_agents + 2 * self.n_objects > self.grid_width * self.grid_length
        ):
            raise ValueError("Grid must fit n_agents + 2 * n_objects distinct starting cells.")
        self.initial_state = deepcopy(initial_state)
        self.debug_mode = debug_mode
        self.create_video = create_video
        self.render_mode = render_mode
        self.renderer = None
        self.frames = []
        self.action_set = {action.value for action in Action}
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.n_agents)

        position_space = spaces.Tuple(
            (spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_length))
        )
        agent_space = spaces.Dict(
            {
                "position": position_space,
                "picker": spaces.Discrete(2),
                "carrying_object": spaces.Discrete(self.n_objects + 1, start=-1),
            }
        )
        object_space = spaces.Dict(
            {"position": position_space, "id": spaces.Discrete(self.n_objects)}
        )
        agent_observation_space = spaces.Dict(
            {
                "self": agent_space,
                "agents": spaces.Tuple([agent_space] * (self.n_agents - 1)),
                "objects": spaces.Tuple([object_space] * self.n_objects),
                "goals": spaces.Tuple([position_space] * self.n_objects),
            }
        )
        self.observation_space = spaces.Dict(
            {f"agent_{i}": agent_observation_space for i in range(self.n_agents)}
        )
        self.reset(seed=seed)

    @staticmethod
    def _positive_int(value, name):
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @property
    def action_space_n(self):
        return len(Action) ** self.n_agents

    @property
    def num_agents(self):
        return self.n_agents

    def _validate_actions(self, actions):
        if isinstance(actions, np.ndarray) and actions.ndim != 1:
            raise ValueError("Actions must be a one-dimensional sequence.")
        if not isinstance(actions, (list, tuple, np.ndarray)) or len(actions) != self.n_agents:
            raise ValueError(f"Expected exactly {self.n_agents} integer actions.")
        if any(not Action.is_valid(action) for action in actions):
            raise ValueError("Actions must be integers from 0 to 5.")

    def get_obs(self):
        return {
            f"agent_{i}": agent.get_agent_obs(self.agents, self.objects, self.goals)
            for i, agent in enumerate(self.agents)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.initial_state is not None:
            self.reset_from_obs(self.initial_state)
        else:
            self.random_reset()
        self.frames = []
        if self.debug_mode:
            self._print_state()
        if self.render_mode == "human":
            self.render()
        return self.get_obs(), {}

    def random_reset(self, seed=None):
        """Reset using only this environment's RNG; optionally reseed it."""
        if self.n_agents + 2 * self.n_objects > self.grid_width * self.grid_length:
            raise ValueError("Grid must fit n_agents + 2 * n_objects distinct starting cells.")
        if seed is not None:
            super().reset(seed=seed)
        cells = self.np_random.permutation(self.grid_width * self.grid_length)
        positions = [
            (int(cell // self.grid_length), int(cell % self.grid_length)) for cell in cells
        ]
        roles = self.np_random.permutation(
            [True] * self.n_pickers + [False] * (self.n_agents - self.n_pickers)
        )
        self.objects = [Object(positions[i], i) for i in range(self.n_objects)]
        offset = self.n_objects
        self.agents = [Agent(positions[offset + i], bool(roles[i])) for i in range(self.n_agents)]
        offset += self.n_agents
        self.goals = positions[offset : offset + self.n_objects]
        self.done = False
        self.frames = []

    def _validate_position(self, position):
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) for v in position)
            or not 0 <= position[0] < self.grid_width
            or not 0 <= position[1] < self.grid_length
        ):
            raise ValueError(f"Invalid grid position: {position!r}")
        return tuple(int(v) for v in position)

    def reset_from_obs(self, obs):
        """Load a validated global state, not a per-agent observation dictionary.

        Agent keys must be agent_0, ..., agent_N; input dictionary order does not
        affect identity. None or -1 represents an empty hand in this input format.
        Validation finishes before the live state is replaced.
        """
        if not isinstance(obs, dict) or not {"agents", "objects", "goals"} <= obs.keys():
            raise ValueError("State must contain agents, objects and goals.")
        agent_states, object_states, goal_states = obs["agents"], obs["objects"], obs["goals"]
        if not isinstance(agent_states, dict) or set(agent_states) != {
            f"agent_{i}" for i in range(self.n_agents)
        }:
            raise ValueError("State must contain exactly agent_0 through agent_{n_agents-1}.")
        if not isinstance(object_states, (list, tuple)) or len(object_states) != self.n_objects:
            raise ValueError("State object count must match n_objects.")
        if not isinstance(goal_states, (list, tuple)) or len(goal_states) != self.n_objects:
            raise ValueError("There must be exactly one goal per object.")
        try:
            object_ids = [obj["id"] for obj in object_states]
            if any(
                not isinstance(i, Integral) or isinstance(i, (bool, np.bool_)) for i in object_ids
            ) or sorted(object_ids) != list(range(self.n_objects)):
                raise ValueError("Object IDs must be unique integers from 0 to n_objects - 1.")
            objects = {
                int(obj["id"]): Object(self._validate_position(obj["position"]), int(obj["id"]))
                for obj in object_states
            }
            goals = [self._validate_position(goal) for goal in goal_states]
            if len(set(goals)) != len(goals):
                raise ValueError("Goal positions must be distinct.")
            agents = []
            for i in range(self.n_agents):
                state = agent_states[f"agent_{i}"]
                position = self._validate_position(state["position"])
                picker = state["picker"]
                if not isinstance(picker, (bool, np.bool_, Integral)) or picker not in (0, 1):
                    raise ValueError("Picker flags must be booleans or 0/1.")
                object_id = state["carrying_object"]
                if object_id is not None and (
                    not isinstance(object_id, Integral)
                    or isinstance(object_id, (bool, np.bool_))
                    or object_id not in range(-1, self.n_objects)
                ):
                    raise ValueError("Carried object ID must be None, -1 or a valid object ID.")
                obj = None if object_id is None or object_id == -1 else objects[object_id]
                if obj is not None and obj.position != position:
                    raise ValueError("A carried object's position must match its carrier.")
                agents.append(Agent(position, bool(picker), obj))
            if sum(agent.picker for agent in agents) != self.n_pickers:
                raise ValueError("State picker count must match n_pickers.")
            if len({agent.position for agent in agents}) != self.n_agents:
                raise ValueError("Agent positions must be distinct.")
            if len({obj.position for obj in objects.values()}) != self.n_objects:
                raise ValueError("Object positions must be distinct.")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed state entry.") from exc
        self.agents = agents
        self.objects = [objects[i] for i in range(self.n_objects)]
        self.goals = goals
        self.done = False
        self.frames = []

    def step(self, actions):
        if self.done:
            raise RuntimeError("Episode has ended; call reset() before stepping again.")
        self._validate_actions(actions)
        for agent in self.agents:
            agent.reward = REWARD_STEP
        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_passes(actions)
        self._handle_drops()
        self.done = self.check_termination()
        if self.done:
            for agent in self.agents:
                agent.reward += REWARD_COMPLETION
        if self.create_video:
            self.frames.append(self._render_frame("rgb_array"))
        if self.render_mode == "human":
            self.render()
        if self.debug_mode:
            self._print_state()
        return self.get_obs(), sum(self._get_rewards()), self.done, False, {}

    def _get_rewards(self):
        return [agent.reward for agent in self.agents]

    def _move_agent(self, agent, action):
        x, y = agent.position
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.get(action, (0, 0))
        position = (
            min(max(x + dx, 0), self.grid_width - 1),
            min(max(y + dy, 0), self.grid_length - 1),
        )
        if any(other is not agent and other.position == position for other in self.agents):
            return agent.position
        if agent.carrying_object is not None and any(
            obj is not agent.carrying_object and obj.position == position for obj in self.objects
        ):
            return agent.position
        agent.position = position
        return position

    def _handle_moves(self, actions):
        for agent, action in zip(self.agents, actions):
            self._move_agent(agent, action)

    def _handle_pickups(self):
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if (
                        obj.carrying_agent is None
                        and obj.position == agent.position
                        and obj.position not in self.goals
                    ):
                        agent.pick_up(obj)
                        break

    def _handle_drops(self):
        for agent in self.agents:
            if (
                agent.carrying_object is not None
                and not agent.picker
                and agent.position in self.goals
            ):
                if not any(
                    obj is not agent.carrying_object and obj.position == agent.position
                    for obj in self.objects
                ):
                    agent.drop(agent.carrying_object)

    def _can_receive_object(self, giver, giver_action, receiver, receiver_action):
        return (
            giver is not receiver
            and giver.carrying_object is not None
            and giver_action == receiver_action == Action.PASS.value
            and receiver.carrying_object is None
            and receiver.position in self._get_adjacent_positions(giver.position)
            and not any(obj.position == receiver.position for obj in self.objects)
        )

    def _handle_passes(self, actions):
        # Rank all candidates BEFORE reserving participants. Stable iteration
        # breaks ties by giver index, then receiver index. Each participates once.
        candidates = [
            (giver, receiver)
            for i, giver in enumerate(self.agents)
            for j, receiver in enumerate(self.agents)
            if self._can_receive_object(giver, actions[i], receiver, actions[j])
        ]
        candidates.sort(key=lambda pair: not (pair[0].picker and not pair[1].picker))
        involved = set()
        for giver, receiver in candidates:
            if giver in involved or receiver in involved:
                continue
            giver.pass_object(receiver)
            self._reward_agents(giver, receiver)
            involved.update((giver, receiver))

    def _reward_agents(self, giver, receiver):
        reward = (
            REWARD_GOOD_PASS
            if giver.picker and not receiver.picker
            else (REWARD_BAD_PASS if not giver.picker and receiver.picker else 0)
        )
        giver.reward += reward
        receiver.reward += reward

    def _get_adjacent_positions(self, position):
        x, y = position
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def check_termination(self):
        return all(
            obj.position in self.goals and obj.carrying_agent is None for obj in self.objects
        )

    def _print_state(self):
        print(self.get_obs())
        print("Agent rewards:", self._get_rewards())

    @staticmethod
    def obs_to_hash(obs):
        """Stable state key for the bundled tabular agents."""
        return hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest()

    def _render_frame(self, mode):
        if self.renderer is None:
            from macpp.core.rendering import Viewer

            self.renderer = Viewer(self)
        return self.renderer.render(mode)

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_frame(self.render_mode)

    def save_video(self, filename, fps=None):
        """Save frames captured with create_video=True; requires the video extra."""
        if not self.frames:
            raise ValueError("No frames to save; step with create_video=True first.")
        import imageio.v2 as imageio

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            path, self.frames, fps=fps or self.metadata["render_fps"], macro_block_size=1
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_action_space_size(self):
        return self.action_space_n

    def _get_state_space_size(self):
        """Legacy combinatorial upper bound, including physically invalid states."""
        cells = self.grid_width * self.grid_length
        agent_states = cells * 2 * (self.n_objects + 1)
        object_states = cells * self.n_objects
        return agent_states**self.n_agents * object_states**self.n_objects * cells**self.n_objects


def make_env(
    width: int, length: int, n_agents: int, n_pickers: int, n_objects: Optional[int] = None
) -> Callable[[], MACPPEnv]:
    def _init():
        return MACPPEnv((width, length), n_agents, n_pickers, n_objects)

    return _init
