from __future__ import annotations
from typing import TYPE_CHECKING
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
class DenseReward:
    '''v0 - every step a reward to incentives easy learning'''
    REWARD_STEP = -1
    REWARD_GOOD_PASS = 5
    REWARD_BAD_PASS = -10
    REWARD_DROP = 10
    REWARD_PICKUP = 10
    REWARD_COMPLETION = 20

class TakeTimeReward:
    '''v2 - original rewards but without incentive to finish early'''
    REWARD_STEP = 0
    REWARD_GOOD_PASS = 5
    REWARD_BAD_PASS = -10
    REWARD_DROP = 10
    REWARD_PICKUP = 10
    REWARD_COMPLETION = 20

class StandardisedReward:
    ''' v3 - Standardised rewards '''
    REWARD_STEP = -0.1
    REWARD_GOOD_PASS = 0.5
    REWARD_BAD_PASS = -1
    REWARD_DROP = 1
    REWARD_PICKUP = 1
    REWARD_COMPLETION = 1


class StandardisedReward2:
    ''' v4 Standardised rewards '''
    REWARD_STEP = -0.1
    REWARD_GOOD_PASS = 0.5
    REWARD_BAD_PASS = -1
    REWARD_DROP = 1
    REWARD_PICKUP = 1
    REWARD_COMPLETION = 0

    
class StandardisedReward3:
    ''' v5
    Standardised - time steps + completion rewards '''
    REWARD_STEP = 0
    REWARD_GOOD_PASS = 0.5
    REWARD_BAD_PASS = -1
    REWARD_DROP = 1
    REWARD_PICKUP = 1
    REWARD_COMPLETION = 1


class SparseReward:
    ''' v1 - Standardised rewards without incentive to finish early '''
    REWARD_STEP = 0
    REWARD_GOOD_PASS = 0.5
    REWARD_BAD_PASS = -1
    REWARD_DROP = 1
    REWARD_PICKUP = 1
    REWARD_COMPLETION = 0


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PASS = 4
    WAIT = 5

    @staticmethod
    def is_valid(action):
        return action in Action._value2member_map_


class Object:
    """
    Represents an object in the environment.

    Attributes:
        position (Tuple[int, int]): The current position of the object.
        id (int): A unique identifier for the object.
        carrying_agent (Optional['Agent']): The agent carrying the object, or None if not carried.

    Methods:
        get_object_obs() -> Dict[str, Union[Tuple[int, int], int]]: Returns object observation.
    """
    def __init__(self,
                 position: Tuple[int, int],
                 id: int):
        self._position = position
        self.id = id
        self._carrying_agent = None

    def get_object_obs(self) -> Dict[str, Union[Tuple[int, int], int]]:
        return {"id": self.id, "position": self.position}

    @property
    def position(self) -> Tuple[int, int]:
        if self._carrying_agent:
            return self._carrying_agent.position
        return self._position

    @position.setter
    def position(self, value: Tuple[int, int]) -> None:
        self._position = value

    @property
    def carrying_agent(self) -> Optional['Agent']:
        return self._carrying_agent

    @carrying_agent.setter
    def carrying_agent(self, agent: Optional['Agent']) -> None:
        self._carrying_agent = agent

class Agent:
    """
    Represents an agent in the environment.

    Attributes:
        position (Tuple[int, int]): The current position of the agent.
        picker (bool): True if the agent is a picker, False otherwise.
        carrying_object (Optional['Object']): The object the agent is carrying, or None if not carrying.
        reward (int): The reward earned by the agent.

    Methods:
        move_up(), move_down(grid_length), move_left(), move_right(grid_width): Move the agent.
        pick_up(obj: Object): Pick up an object if the agent is a picker and not carrying.
        drop(obj: Object): Drop an object if the agent is not a picker and carrying.
        pass_object(other_agent: 'Agent'): Pass the carried object to another agent.
        get_agent_obs(all_agents: list['Agent'], all_objects: list[Object], goals: Any) -> Dict[str, Any]: 
            Get the agent's observation.
        get_basic_agent_obs() -> Dict[str, Any]: Get a basic observation of the agent.
    """
    carrying_object: Optional[Object] = None

    def __init__(self, 
                 position: Tuple[int, int], 
                 picker: bool,
                 reward_class: object,
                 carrying_object: Optional[Object] = None,
                 reward: int = 0,

                 ) -> None:
        self._position = position
        self.picker = picker
        self.carrying_object = carrying_object
        self.reward = reward
        self.reward_class = reward_class

    @property
    def position(self) -> Tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value: Tuple[int, int]) -> None:
        self._position = value
        if self.carrying_object:
            self.carrying_object.position = value

    def move_up(self) -> None:
        x, y = self.position
        self.position = (x, max(0, y - 1))

    def move_down(self, grid_length: int) -> None:
        x, y = self.position
        self.position = (x, min(grid_length - 1, y + 1))

    def move_left(self) -> None:
        x, y = self.position
        self.position = (max(0, x - 1), y)

    def move_right(self, grid_width: int) -> None:
        x, y = self.position
        self.position = (min(grid_width - 1, x + 1), y)

    def pick_up(self, obj: Object) -> None:
        if self.picker and self.carrying_object is None:
            self.carrying_object = obj  
            obj.carrying_agent = self
            self.reward += self.reward_class.REWARD_PICKUP

    def drop(self, obj: Object) -> None:
        if self.carrying_object is not None and not self.picker:
            obj._position = self.position
            obj.carrying_agent = None  
            self.carrying_object = None
            self.reward += self.reward_class.REWARD_DROP

    def pass_object(self, other_agent: 'Agent') -> None:
        if self.carrying_object is not None:
            other_agent.carrying_object = self.carrying_object
            self.carrying_object.carrying_agent = other_agent
            self.carrying_object = None

    def get_agent_obs(self, all_agents: list['Agent'], all_objects: list[Object], goals: Any) -> Dict[str, Any]:
        if not isinstance(self.carrying_object, (Object, type(None))):
            print(f"Unexpected type for carrying_object: {type(self.carrying_object)} with value {self.carrying_object}")
            
        obs = {
            'self': {
                'position': self.position,
                'picker': self.picker,
                'carrying_object': self.carrying_object.id if self.carrying_object else None
            },
            'agents': [other_agent.get_basic_agent_obs() for other_agent in all_agents if other_agent != self],
            'objects': [obj.get_object_obs() for obj in all_objects],
            'goals': goals
        }
        return obs



    def get_basic_agent_obs(self) -> Dict[str, Any]:
        carrying_object_id = self.carrying_object if isinstance(self.carrying_object, int) else self.carrying_object.id if self.carrying_object else None
        return {
            'position': self.position,
            'picker': self.picker,
            'carrying_object': carrying_object_id
        }

class MACPPEnv(gym.Env):
    """
    Collaborative pick and place environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size: Tuple[int, int],
        n_agents: int,
        n_pickers: int,
        n_objects: Optional[int] = 1,
        initial_state=None,
        cell_size: Optional[int] = 50,
        debug_mode: Optional[bool] = False,
        create_video: Optional[bool] = False,
        sparse_reward: Optional[bool] = False,
        take_time_reward: Optional[bool] = False,
        standardised_reward: Optional[bool] = False,
        completion_reward: Optional[bool] = True,
        enumerate_action: Optional[bool] = True,
        seed: Optional[int] = 1000
    ) -> None:

        """
        Initialize the environment.

        Args:
            grid_size (Tuple[int, int]): Size of the grid.
            n_agents (int): Number of agents.
            n_pickers (int): Number of picker agents.
            n_objects (Optional[int]): Number of objects. Default is 1.
            initial_state: Initial state of the environment.
            cell_size (Optional[int]): Size of grid cells. Default is 300.
            debug_mode (Optional[bool]): Enable debug mode. Default is False.
            create_video (Optional[bool]): Create video frames. Default is False.
            seed (Optional[int]): Random seed for environment initialization.

        Returns:
            None
        """

        self.grid_size = grid_size
        self.grid_width, self.grid_length = grid_size
        self.cell_size = cell_size
        self.n_agents = n_agents
        self.n_pickers = n_pickers
        self.initial_state = initial_state
        self.create_video = create_video
        self.debug_mode = debug_mode
        self.enumerate_action = enumerate_action
        # completion_reward = False

        if sparse_reward:
            self.reward_class = SparseReward()
                          
        elif standardised_reward:
            if take_time_reward:
                self.reward_class = StandardisedReward3()
                print('using stand3 reward')
            else:
                self.reward_class = StandardisedReward()
        elif not completion_reward:
            self.reward_class = StandardisedReward2()
        elif take_time_reward:
            self.reward_class = TakeTimeReward()
        else:
            self.reward_class = DenseReward()
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
                    (spaces.Discrete(self.grid_width),
                     spaces.Discrete(self.grid_length))
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
                    (spaces.Discrete(self.grid_width),
                     spaces.Discrete(self.grid_length))
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
            "agents": spaces.Tuple([agent_space] * (self.n_agents-1)),
            "objects": spaces.Tuple([object_space] * self.n_objects),
            "goals": spaces.Tuple([goal_space] * self.n_objects)
        })

        # The observation space
        self.observation_space = spaces.Dict({
            f"agent_{i}": agent_observation_space for i in range(self.n_agents)
        })


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

        self.done = [False] * self.n_agents
        # Initialise the environment either randomly or from a state
        if initial_state is None:
            self.random_reset()
        else:
            self.reset_from_obs(initial_state)

        # Rendering
        self._rendering_initialised = False
        self.renderer = None

        # If a video is required, create frames
        if self.create_video and self.cell_size is not None:
            self.offscreen_surface = pygame.Surface(
                (self.grid_width * self.cell_size,
                 self.grid_length * self.cell_size)
            )
            self.frames = []

    # @property
    # def action_space_n(self):
    #     return np.prod(self.action_space.nvec)

    @property
    def action_space_n(self):
        return np.prod([len(Action)] * self.n_agents)

    @property
    def num_agents(self):
        return self.n_agents

    def _validate_actions(self, actions: List[int]) -> None:
        for action in actions:
            if isinstance(action, np.ndarray):
                action = np.argmax(action)
            if action is None or not (00 <= action <= len(Action)-1):
                raise ValueError(
                    f"Invalid action: {action}.")

    def get_obs(self) -> Dict[str, Dict[str, Any]]:
        observations = {}
        for idx, agent in enumerate(self.agents):
            observations[f"agent_{idx}"] = agent.get_agent_obs(
                self.agents, self.objects, self.goals)
        return observations

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (Optional[int]): Random seed for environment initialization.
            options (Optional[Any]): Additional options for resetting.

        Returns:
            Tuple containing initial observations and additional info.
        """
        if self.initial_state:
            self.reset_from_obs(self.initial_state)
        else:
            self.random_reset(seed)

        self.done = [False] * self.n_agents
        obs = self.get_obs()
        
        # Debug info
        if self.debug_mode:
            print("State from reset:")
            self._print_state()

        return obs

    def random_reset(self, seed: Optional[int] = None) -> None:
        """
        Initialise the environment with random allocations of agents and objects
        """

        if self.debug_mode:
            print("Random reset.")
        rng = np.random.default_rng(seed)
        all_positions = [(x, y) for x in range(self.grid_width)
                         for y in range(self.grid_length)]
        rng.shuffle(all_positions)

        # Randomly assign Picker flags to agents
        picker_flags = [True] * self.n_pickers + [False] * (
            self.n_agents - self.n_pickers
        )
        # random.shuffle(picker_flags)

        # Randomly allocate object positions
        object_positions = random.sample(all_positions, self.n_objects)
        for obj_pos in object_positions:
            all_positions.remove(obj_pos)
        # print(object_positions)

        # Randomly allocate agent positions
        agent_positions = random.sample(all_positions, self.n_agents)
        for agent_pos in agent_positions:
            all_positions.remove(agent_pos)

        # Randomly allocate goal positions.
        goal_positions = random.sample(all_positions, self.n_objects)

        # object_positions = [(3,1)]
        # goal_positions = [(4,1)]
        # agent_positions = [(0,1), (1,3)]


        # Initialize agents
        self.agents = []
        for _ in range(self.n_agents):
            agent_position = agent_positions.pop()
            self.agents.append(
                Agent(
                    position=agent_position,
                    picker=picker_flags.pop(),
                    reward_class=self.reward_class,
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

    def reset_from_obs(self, obs: Dict[str, Dict[str, Any]]) -> None:
        """
        Reset the environment to a predefined initial state.
        """

        if self.debug_mode:
            print("Reset from state.")

        agent_states = obs['agents']
        object_states = obs['objects']
        goal_states = obs['goals']

        # Initialize objects first
        self.objects = []
        for object_state in object_states:
            obj = Object(
                position=tuple(object_state['position']),
                id=object_state['id']
            )
            self.objects.append(obj)

        # Reset agents
        self.agents = []
        for agent_state in agent_states.values():
            # If the agent is carrying an object, find and assign it
            carrying_object = None
            if agent_state['carrying_object'] is not None:
                for obj in self.objects:
                    if obj.id == agent_state['carrying_object']:
                        carrying_object = obj
                        break

            agent = Agent(
                position=tuple(agent_state['position']),
                picker=agent_state['picker'],
                reward_class=self.reward_class,
                carrying_object=carrying_object,  
                reward=0  
            )
            self.agents.append(agent)

            # Now set the carrying agent properly
            if carrying_object:
                carrying_object.carrying_agent = agent

        # Reset goals
        self.goals = [tuple(goal) for goal in goal_states]

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
                f"Picker: {str(agent.picker):<5}, Status: {carrying_status:<12}, {carrying_object}, "
                f"Reward: {agent.reward}"
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
        """
        Perform one time step in the environment.

        Args:
            actions (List[int]): List of actions for each agent.

        Returns:
            Tuple containing observations, total reward, termination flag, and additional info.
        """
        # Check that no invalid actions are taken
        if self.debug_mode:
            self._validate_actions(actions)
            print(f"\nExecuting actions: {actions}\n")

        # Negative reward given at every step
        for agent in self.agents:
            agent.reward = self.reward_class.REWARD_STEP
            if self.debug_mode:
                print(f'Rewarded for step: {self.reward_class.REWARD_STEP}')

        # Execute the actions
        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_passes(actions)
        self._handle_drops()

        # Check for termination
        if self.check_termination():
            for agent in self.agents:
                agent.reward += self.reward_class.REWARD_COMPLETION
                if self.debug_mode:
                    print(f'Rewarded for completion: {self.reward_class.REWARD_COMPLETION}')
            self.done = [True] * self.n_agents

        # Collect frames for the video when required
        if self.create_video:
            self.frames.append(
                pygame.surface.array3d(self.offscreen_surface))

        # Debug info
        if self.debug_mode:
            self._print_state()

        # total_reward = sum(self._get_rewards())
        total_reward = self._get_rewards()
        if self.debug_mode:
            print(f'Total reward: {total_reward}')
            for idx, agent in enumerate(self.agents):
                print(f"Agent {idx} Reward: {agent.reward}")

        obs = self.get_obs()

        return obs, total_reward, self.done, {}

    def _get_rewards(self) -> List[int]:
        return [[agent.reward] for agent in self.agents] #FIXME I have put a list around this because it did not work with SubProcVec otherwise

    def _move_agent(self, agent: Agent, action: int) -> Tuple[int, int]:
        """
        Move an agent based on the specified action.

        Args:
            agent (Agent): The agent to move.
            action (int): The action representing the direction of movement.

        Returns:
            Tuple[int, int]: The new position of the agent after the move.
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

        # If the new position contains an agent, move is not allowed
        if any(other_agent.position == new_position for other_agent in self.agents if other_agent != agent):
            return agent.position

        # If the agent is carrying an object, and the new position contains an object, move is not allowed
        if agent.carrying_object is not None and any(obj.position == new_position for obj in self.objects):
            return agent.position

        agent.position = new_position
        return new_position

    def _handle_moves(self, actions: List[int]) -> None:
        for idx, agent_action in enumerate(actions):
            agent = self.agents[idx]
            if isinstance(agent_action, np.ndarray):
                agent_action = np.argmax(agent_action)
            self._move_agent(agent, agent_action)

            # If the agent is carrying an object, update the object's position
            if agent.carrying_object is not None:
                carried_obj = next((obj for obj in self.objects if obj.id == agent.carrying_object), None)
                if carried_obj:
                    carried_obj.carrying_agent = agent

    def _handle_pickups(self) -> None:
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position in self.goals:
                        continue
                        # TODO still needed?
                    if obj.position == agent.position and not obj.carrying_agent:
                        agent.pick_up(obj) 

                        if self.debug_mode:
                            print(f'Rewarded for pickup: {self.reward_class.REWARD_PICKUP}')
                        break

    def _handle_drops(self) -> None:
        for agent in self.agents:
            if agent.carrying_object is not None and not agent.picker:
                if agent.position in self.goals:
                    # Check if the goal position already has an object
                    if not any(obj.position == agent.position and obj != agent.carrying_object for obj in self.objects):
                        agent.carrying_object.position = agent.position
                        agent.carrying_object.carrying_agent = None  
                        agent.carrying_object = None
                        agent.reward += self.reward_class.REWARD_DROP
                        if self.debug_mode:
                            print(f'Rewarded for dropoff: {self.reward_class.REWARD_DROP}')

    from typing import List, Tuple

    def _handle_passes(self, actions: List[int]) -> None:
        """
        Handle simultaneous object passes between agents based on their actions and positions.

        The function plans all valid passes before executing any to ensure consistency in agent states.
        It prioritizes passes from picker agents to non-picker agents.

        Args:
        actions (List[int]): A list of actions corresponding to each agent.
        """
        # Identify eligible givers and receivers
        actions = [np.argmax(action) for action in actions if isinstance(action, np.ndarray)]

        eligible_givers = [
            (idx, agent) for idx, (agent, action) in enumerate(zip(self.agents, actions))
            if action == Action.PASS.value and agent.carrying_object is not None
        ]
        eligible_receivers = [
            (idx, agent) for idx, (agent, action) in enumerate(zip(self.agents, actions))
            if action == Action.PASS.value and agent.carrying_object is None
        ]

        # Plan the possible passes
        planned_passes = []
        involved_agents = set()
        for (giver_idx, giver) in eligible_givers:
            if giver in involved_agents:
                continue  # Skip if giver is already involved in a planned pass
            for adj_pos in self._get_adjacent_positions(giver.position):
                for (receiver_idx, receiver) in eligible_receivers:
                    if receiver in involved_agents:
                        continue  # Skip if receiver is already involved in a planned pass
                    if receiver.position == adj_pos and giver_idx != receiver_idx:
                        if self._can_receive_object(giver, Action.PASS.value, receiver, Action.PASS.value):
                            planned_passes.append((giver, receiver, giver.picker and not receiver.picker))
                            involved_agents.add(giver)
                            involved_agents.add(receiver)
                            break 

        # Sort planned passes to prioritize picker to non-picker passes
        planned_passes.sort(key=lambda x: x[2], reverse=True)

        # Execute passes
        for giver, receiver, _ in planned_passes:
            if giver.carrying_object is not None and receiver.carrying_object is None:
                giver.pass_object(receiver)
                self._reward_agents(giver, receiver)

    def _can_receive_object(self, giver: Agent, giver_action: int, receiver: Agent, receiver_action: int) -> bool:
        return (
            giver.carrying_object is not None and
            giver_action == Action.PASS.value and
            receiver_action == Action.PASS.value and
            receiver.carrying_object is None and
            not any(obj.position == receiver.position for obj in self.objects)
        )

    def _reward_agents(self, giver: Agent, receiver: Agent) -> None:
        if giver.reward is not None and receiver.reward is not None:
            if giver.picker and not receiver.picker:
                giver.reward += self.reward_class.REWARD_GOOD_PASS
                receiver.reward += self.reward_class.REWARD_GOOD_PASS
                self._log_reward("good", self.reward_class.REWARD_GOOD_PASS)
            elif not giver.picker and receiver.picker:
                giver.reward += self.reward_class.REWARD_BAD_PASS
                receiver.reward += self.reward_class.REWARD_BAD_PASS
                self._log_reward("bad", self.reward_class.REWARD_BAD_PASS)
        else:
            print("Warning: Attempted to update reward of agent with None reward.")


    def _log_reward(self, pass_type: str, reward_amount: int) -> None:
        if self.debug_mode:
            print(f'Rewarded for {pass_type} pass: {reward_amount}')

    def _get_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def check_termination(self) -> bool:
        for obj in self.objects:
            if obj.position not in self.goals or obj.carrying_agent is not None:
                return False
        return True

    def _init_render(self) -> None:
        from macpp.core.rendering import Viewer
        self.renderer = Viewer(self)
        if self.debug_mode:
            print("Rendering initialised.")
        self._rendering_initialised = True

    def render(self, mode: str = 'human', save=False) -> Union[None, np.ndarray]:
        """
        Render the environment.

        Args:
            mode (str): Rendering mode ('human' or other).

        Returns:
            Rendered image or None.
        """
        if not self._rendering_initialised:
            self._init_render()
        return self.renderer.render(save=save)

    def close(self) -> None:
        """
        Close the environment, release resources.
        """
        if self.renderer:
            self.renderer.close()
        # Save the video frames
        if self.create_video:
            pass

    def _get_state_space_size(self) -> int:
        agent_space_size = self.grid_width * \
            self.grid_length * 2 * (self.n_objects + 1)
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
        return MACPPEnv((width, length), n_agents, n_pickers, n_objects)
    return _init

