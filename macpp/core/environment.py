import gym
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from gym import spaces
import random
import pygame
import numpy as np
import time
import sys
import hashlib

# sys.path.append("/home/gm13/Dropbox/mycode/envs/collaborative_pick_and_place/macpp/")
# sys.path.append("/Users/giovannimontana/Dropbox/mycode/envs/collaborative_pick_and_place/macpp")


# Environment's rewards
REWARD_STEP = -1
REWARD_GOOD_PASS = 5
REWARD_BAD_PASS = -5
REWARD_DROP = 10
REWARD_COMPLETION = 20


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PASS = 4
    WAIT = 5


class Agent:
    def __init__(self,
                 position: Tuple[int, int],
                 picker: bool,
                 carrying_object: Optional[int] = None,
                 reward: Optional[int] =0) -> None:
        self.position = position
        self.picker = picker
        self.carrying_object = carrying_object
        self.reward = reward

    def get_agent_obs(self, all_agents, all_objects, goals) -> Dict[str, Any]:
        obs = {
            'self': {
                'position': self.position,
                'picker': self.picker,
                'carrying_object': self.carrying_object
            },
            'agents': [other_agent.get_basic_agent_obs() for other_agent in all_agents if other_agent != self],
            'objects': [obj.get_object_obs() for obj in all_objects],
            'goals': goals
        }
        return obs

    def get_basic_agent_obs(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'picker': self.picker,
            'carrying_object': self.carrying_object
        }

class Object:
    def __init__(self,
                 position: Tuple[int, int],
                 id: int) -> None:
        self.position = position
        self.id = id

    def get_object_obs(self) -> Dict[str, Union[Tuple[int, int], int]]:
        return {"id": self.id, "position": self.position}


class MACPPEnv(gym.Env):
    """
    Class implementing the logic for the collaborative pick and place environment
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size: Tuple[int, int],
        n_agents: int,
        n_pickers: int,
        n_objects: Optional[int] = 1,
        initial_state=None,
        cell_size: Optional[int] = 300,
        debug_mode: Optional[bool] = False,
        create_video: Optional[bool] = False,
        seed: Optional[int] = None
    ) -> None:

        # Dimesions of the grid and cell size
        self.grid_width = grid_size[0]
        self.grid_length = grid_size[1]
        self.cell_size = cell_size
        self.n_agents = n_agents
        self.n_pickers = n_pickers
        self.initial_state = initial_state
        self.create_video = create_video
        self.debug_mode = debug_mode

        # Check that there are at least two agents
        if n_agents < 2:
            raise ValueError(
                "Invalid number of agents. There should be at least two agents.")

        # Check if there are enough pickers and that n_pickers is not the same or larger than n_agents
        if n_pickers <= 0 or n_pickers >= n_agents:
            raise ValueError(
                "Invalid number of pickers. There should be at least one picker and the number of pickers should be less than the total number of agents.")

        # Set the number of objects and goals
        if n_objects is None:
            self.n_objects = self.n_agents
        else:
            self.n_objects = n_objects

        # Check if the grid size is sufficiently large
        total_cells = self.grid_width * self.grid_length
        total_entities = self.n_agents + self.n_objects
        if total_entities > total_cells:
            raise ValueError(
                "Grid size not sufficiently large to contain all the entities."
            )

        # The action space
        self.action_set = set(action.value for action in Action)
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.n_agents)

        # An agent's observation space
        agent_space = spaces.Dict(
            {
                "position": spaces.Tuple(
                    (spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_length))
                ),
                "picker": spaces.Discrete(2),  # 0 or 1
                "carrying_object": spaces.Discrete(
                    self.n_objects + 1
                ),  # Including a value for "not carrying"
            }
        )

        # An object's observation space
        object_space = spaces.Dict(
            {
                "position": spaces.Tuple(
                    (spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_length))
                ),
                "id": spaces.Discrete(self.n_objects),
            }
        )

        # A goal's observation space
        goal_space = spaces.Tuple(
            (spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_length))
        )

        # An agent's observation space
        agent_observation_space = spaces.Dict({
            "self": agent_space,
            "agents": spaces.Tuple([agent_space] * (self.n_objects-1)),
            "objects": spaces.Tuple([object_space] * self.n_objects),
            "goals": spaces.Tuple([goal_space] * self.n_objects)
        })

        # The observation space
        self.observation_space = spaces.Dict({
            f"agent_{i}": agent_observation_space for i in range(self.n_agents)
        })

        # self.observation_space = spaces.Dict(
        #     {
        #         "agents": spaces.Tuple([self.agent_space] * self.n_agents),
        #         "objects": spaces.Tuple([self.object_space] * self.n_objects),
        #         "goals": spaces.Tuple([self.goal_space] * self.n_objects),
        #     }
        # )

        self.done = False

        # Initialise the environment either randomly or from a state
        if initial_state is None:
            self.random_reset()
        else:
            self.reset_from_obs(initial_state)

        # Rendering
        self._rendering_initialised = False
        self.renderer = None

        # If a video is required, create frames
        if self.create_video:
            self.offscreen_surface = pygame.Surface(
                (self.grid_width * self.cell_size, self.grid_length * self.cell_size)
            )
            self.frames = []

    def _validate_actions(self, actions: List[int]) -> None:
        for action in actions:
            if action is None or not (00 <= action <= len(Action)-1):
                raise ValueError(
                    f"Invalid action: {action}.")

    def get_obs(self) -> Dict[str, Dict[str, Any]]:
        observations = {}
        for idx, agent in enumerate(self.agents):
            observations[f"agent_{idx}"] = agent.get_agent_obs(self.agents, self.objects, self.goals)
        return observations

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Reset the environment to either a random state or an predefined initial state
        """
        if self.initial_state:
            observations = self.reset_from_obs(self.initial_state)
        else:
            observations = self.random_reset(seed)

        self.done = [False for _ in range(self.n_agents)]

        return observation, {}


    def obs_to_hash(self, obs: Dict[str, Dict[str, Any]]) -> str:
        concatenated_obs = ''.join([str(obs[agent_id]) for agent_id in sorted(obs.keys())])
        return hashlib.md5(concatenated_obs.encode()).hexdigest()

    def random_reset(self, seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Initialise the environment with random allocations of agents and objects
        """
        rng = np.random.default_rng(seed)
        all_positions = [(x, y) for x in range(self.grid_width)
                         for y in range(self.grid_length)]
        rng.shuffle(all_positions)

        # Randomly assign Picker flags to agents
        picker_flags = [True] * self.n_pickers + [False] * (
            self.n_agents - self.n_pickers
        )
        random.shuffle(picker_flags)

        # Randomly allocate object positions
        object_positions = random.sample(all_positions, self.n_objects)
        for obj_pos in object_positions:
            all_positions.remove(obj_pos)

        # Randomly allocate agent positions
        agent_positions = random.sample(all_positions, self.n_agents)
        for agent_pos in agent_positions:
            all_positions.remove(agent_pos)

        # Randomly allocate goal positions.
        goal_positions = random.sample(all_positions, self.n_objects)

        # Initialize agents
        self.agents = []
        for _ in range(self.n_agents):
            agent_position = agent_positions.pop()
            self.agents.append(
                Agent(
                    position=agent_position,
                    picker=picker_flags.pop(),
                    carrying_object=None,
                    reward=0
                )
            )

        # Initialize objects
        self.objects = [
            Object(position=obj_pos, id=i) for i, obj_pos in enumerate(object_positions)
        ]

        # Assign goals
        self.goals = goal_positions

        return self.get_obs()

    def reset_from_obs(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Reset the environment to a predefined initial state.
        """
        agent_states = obs['agents']
        object_states = obs['objects']
        goal_states = obs['goals']

        # Reset agents
        self.agents = []
        for agent_state in agent_states.values():
            agent = Agent(
                position=tuple(agent_state['position']),
                picker=agent_state['picker'],
                carrying_object=agent_state['carrying_object'],
                reward=0  # Reset reward to 0
            )
            self.agents.append(agent)

        # Reset objects
        self.objects = []
        for object_state in object_states:
            obj = Object(
                position=tuple(object_state['position']),
                id=object_state['id']
            )
            self.objects.append(obj)

        # Reset goals
        self.goals = [tuple(goal) for goal in goal_states]

        return self.get_obs()

    def _print_state(self):
        print("-" * 30)
        for idx, agent in enumerate(self.agents, start=1):
            carrying_status = (
                "Carrying" if agent.carrying_object is not None else "Not Carrying"
            )
            carrying_object = (
                f"Object ID: {agent.carrying_object}"
                if agent.carrying_object is not None
                else "None"
            )
            print(
                f"- Agent {idx:<2}: Position: {agent.position}, "
                f"Picker: {str(agent.picker):<5}, Status: {carrying_status:<12}, {carrying_object}"
            )

        for idx, obj in enumerate(self.objects, start=1):
            print(f"- Object {idx}: Position: {obj.position}, ID: {obj.id}")

        if not self.goals:
            print("- No goal positions set.")
        else:
            for idx, goal in enumerate(self.goals):
                print(f"- Goal {idx + 1}: Position {goal}")
        print("-" * 30)

    def _random_position(self):
        return (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_length - 1))

    def step(self, actions: List[int]) -> Tuple[Dict[str, Dict[str, Any]], int, bool, Dict[str, Any]]:
        # Check that no invalid actions are taken
        if self.debug_mode:
            self._validate_actions(actions)
            print(f"\nExecuting actions: {actions}\n")

        # Negative reward given at every step
        for agent in self.agents:
            agent.rewards = REWARD_STEP

        # Execute the actions
        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_passes(actions)

        # Check for termination
        if self.check_termination():
            for agent in self.agents:
                agent.reward += REWARD_COMPLETION
            self.done = True

        # Collect frames for the video when required
        if self.create_video:
            self.frames.append(
                pygame.surface.array3d(self.offscreen_surface))

        # Debug info
        if self.debug_mode:
            self._print_state()

        return self.get_obs(), sum(self._get_rewards()), self.done, {}

    def _get_rewards(self) -> List[int]:
        return [agent.rewards for agent in self.agents]

    def _move_agent(self, agent: Agent, action: int) -> Tuple[int, int]:
        """
        Move the envirnment in the grid. Collisions between agents are not allowed.
        """

        x, y = agent.position
        if action == Action.UP.value:
            y = max(0, y - 1)
        elif action == Action.DOWN.value:
            y = min(self.grid_length - 1, y + 1)
        elif action == Action.LEFT.value:
            x = max(0, x - 1)
        elif action == Action.RIGHT.value:
            x = min(self.grid_width - 1, x + 1)

        new_position = (x, y)

        # If agents collide, they don't move
        if new_position not in [a.position for a in self.agents]:
            return new_position
        return agent.position

    def _handle_moves(self, actions: List[int]) -> None:
        for idx, agent_action in enumerate(actions):
            agent = self.agents[idx]
            agent.position = self._move_agent(agent, agent_action)

            # if an agent is carrying an object, the position of the object being carried needs to be updated
            if agent.carrying_object is not None:
                carried_object = next(
                    (o for o in self.objects if o.id == agent.carrying_object), None
                )
                if carried_object:
                    carried_object.position = agent.position

    def _handle_pickups(self) -> None:
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying_object = obj.id
                        break

    def _handle_passes(self, actions: List[int]) -> None:

        # Create a list to store agents that will receive objects
        receiving_agents = [None] * len(self.agents)

        for idx, agent in enumerate(self.agents):
            if actions[idx] == Action.PASS.value and agent.carrying_object is not None:
                x, y = agent.position
                adjacent_positions = [
                    (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for adj_pos in adjacent_positions:
                    adj_agent = next(
                        (a for a in self.agents if a.position == adj_pos), None
                    )
                    if (
                        adj_agent
                        and actions[self.agents.index(adj_agent)] == Action.PASS.value
                        and adj_agent.carrying_object is None
                    ):
                        receiving_agents[
                            self.agents.index(adj_agent)
                        ] = agent.carrying_object
                        obj = next(
                            (o for o in self.objects if o.id ==
                             agent.carrying_object),
                            None,
                        )
                        if obj:
                            obj.position = (
                                adj_agent.position
                            )  # Update the object's position
                        agent.carrying_object = None

                        # Assign rewards based on the type of pass
                        if agent.picker and not adj_agent.picker:
                            agent.reward += REWARD_GOOD_PASS
                            adj_agent.reward += REWARD_GOOD_PASS
                        elif not agent.picker and adj_agent.picker:
                            agent.reward += REWARD_BAD_PASS
                            adj_agent.reward += REWARD_BAD_PASS

                        break

        # Process the passing of objects
        for idx, obj_id in enumerate(receiving_agents):
            if obj_id is not None:
                self.agents[idx].carrying_object = obj_id

    def check_termination(self) -> bool:
        goal_positions = set(self.goals)
        object_positions = {obj.position for obj in self.objects}
        carrying_agents = {
            agent.position for agent in self.agents if agent.carrying_object is not None}
        # Check if every goal has an object on it, no goal has more than one object,
        # and no non-picker agent is carrying an object
        if (object_positions == goal_positions and
            not carrying_agents.intersection(goal_positions) and
                all(agent.picker or agent.carrying_object is None for agent in self.agents)):
            if self.debug_mode:
                print("Termination checked!")
            return True
        return False

    def _handle_drops(self) -> None:
        for agent in self.agents:
            # Check if the agent is carrying an object and is NOT a picker
            if agent.carrying_object is not None and not agent.picker:
                # Check if the agent's position matches any of the goal positions
                if agent.position in self.goals:
                    # Check if the goal position already has an object
                    if not any(obj.position == agent.position and obj.id != agent.carrying_object for obj in self.objects):
                        # Drop the object at the goal position
                        obj = next(
                            obj for obj in self.objects if obj.id == agent.carrying_object)
                        obj.position = agent.position
                        agent.carrying_object = None

    def _init_render(self) -> None:
        from core.rendering import Viewer
        self.renderer = Viewer(self)
        if self.debug_mode:
            print("Rendering initialised.")
        self._rendering_initialised = True

    def render(self, mode: str = 'human') -> Union[None, np.ndarray]:
        if not self._rendering_initialised:
            self._init_render()
        return self.renderer.render()

    def close(self) -> None:
        # Close the renderer
        if self.renderer:
            self.renderer.close()
        # Save the video frames
        if self.create_video:
            pass

    def _get_state_space_size(self) -> int:
        agent_space_size = self.grid_width * self.grid_length * 2 * (self.n_objects + 1)
        object_space_size = self.grid_width * self.grid_length * self.n_objects
        goal_space_size = self.grid_width * self.grid_length
        state_space_size = (agent_space_size ** self.n_agents) * (
            object_space_size ** self.n_objects) * (goal_space_size ** self.n_objects)
        return state_space_size

    def _get_action_space_size(self) -> int:
        single_agent_action_space = len(Action)
        action_space_size = single_agent_action_space ** self.n_agents
        return action_space_size


def make_env(width: int, length: int, n_agents: int, n_pickers: int, n_objects: Optional[int] = None) -> Callable[[], MACPPEnv]:
    def _init():
        return MACPPEnv(width, length, n_agents, n_pickers, n_objects)
    return _init


def game_loop(env, render):
    """
    Game loop for the MultiAgentPickAndPlace environment.
    """
    env.reset()
    done = False

    if render:
        env.render()

    while not done:
        # Sample random actions for each agent
        actions = env.action_space.sample().tolist()
        print(actions)

        nobs, nreward, ndone, _ = env.step(actions)
        print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)

    env.close()
    print("Episode finished.")

