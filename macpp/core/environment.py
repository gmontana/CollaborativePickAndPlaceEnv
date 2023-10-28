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
        return action in Action._value2member_map_


class Agent:
    def __init__(self,
                 position: Tuple[int, int],
                 picker: bool,
                 carrying_object: Optional[int] = None,
                 reward: Optional[int] = 0) -> None:
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

        self.grid_width, self.grid_length = grid_size
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
                (self.grid_width * self.cell_size,
                 self.grid_length * self.cell_size)
            )
            self.frames = []

    @property
    def action_space_n(self):
        return np.prod(self.action_space.nvec)

    @property
    def num_agents(self):
        return self.n_agents

    def _validate_actions(self, actions: List[int]) -> None:
        for action in actions:
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
        Reset the environment to either a random state or an predefined initial state
        """
        if self.initial_state:
            self.reset_from_obs(self.initial_state)
        else:
            self.random_reset(seed)

        self.done = False
        obs = self.get_obs()
        
        # Debug info
        if self.debug_mode:
            print("State from reset:")
            self._print_state()

        return obs, {}

    '''
    def obs_to_hash(self, obs: Dict[str, Dict[str, Any]]) -> str:
        def position_to_number(pos, length):
            return pos[0] * length + pos[1]

        encoded_obs = []

        for agent_id in sorted(obs.keys()):
            agent_data = obs[agent_id]
            # Encode agent's own observation
            encoded_obs.append(position_to_number(agent_data['self']['position'], self.grid_length))
            encoded_obs.append(1 if agent_data['self']['picker'] else 0)
            encoded_obs.append(agent_data['self']['carrying_object'] if agent_data['self']['carrying_object'] is not None else -1)
            # Encode other agents' observations
            for other_agent in agent_data['agents']:
                encoded_obs.append(position_to_number(other_agent['position'], self.grid_length))
                encoded_obs.append(1 if other_agent['picker'] else 0)
                encoded_obs.append(other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)
            # Encode objects' observations
            for obj in agent_data['objects']:
                encoded_obs.append(position_to_number(obj['position'], self.grid_length))
                encoded_obs.append(obj['id'])
            # Encode goals
            for goal in agent_data['goals']:
                encoded_obs.append(position_to_number(goal, self.grid_length))

        # Convert the list of numbers to a single string and hash it
        concatenated_obs = ''.join(map(str, encoded_obs))
        return hashlib.md5(concatenated_obs.encode()).hexdigest()
    '''

    '''
    def obs_to_hash(self, obs: Dict[str, Dict[str, Any]]) -> str:
        def position_to_number(pos, length):
            return pos[0] * length + pos[1]

        encoded_obs = []

        for agent_id in sorted(obs.keys()):
            agent_data = obs[agent_id]
            # Encode agent's own observation
            encoded_obs.append(position_to_number(
                agent_data['self']['position'], self.grid_length))
            encoded_obs.append(1 if agent_data['self']['picker'] else 0)
            encoded_obs.append(agent_data['self']['carrying_object']
                               if agent_data['self']['carrying_object'] is not None else -1)

        # Encode objects' observations
        # We can use 'agent_0' as a reference since all agents see the same objects
        for obj in obs['agent_0']['objects']:
            encoded_obs.append(position_to_number(
                obj['position'], self.grid_length))
            encoded_obs.append(obj['id'])

        # Encode goals
        # We can use 'agent_0' as a reference since all agents see the same goals
        for goal in obs['agent_0']['goals']:
            encoded_obs.append(position_to_number(goal, self.grid_length))

        # Convert the list of numbers to a single string and hash it
        concatenated_obs = ''.join(map(str, encoded_obs))
        return hashlib.md5(concatenated_obs.encode()).hexdigest()
    '''

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

    def reset_from_obs(self, obs: Dict[str, Dict[str, Any]]) -> None:
        """
        Reset the environment to a predefined initial state.
        """

        if self.debug_mode:
            print("Reset from state.")

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
        # Check that no invalid actions are taken
        if self.debug_mode:
            self._validate_actions(actions)
            print(f"\nExecuting actions: {actions}\n")

        # Negative reward given at every step
        for agent in self.agents:
            agent.reward = REWARD_STEP
            if self.debug_mode:
                print(f'Rewarded for step: {REWARD_STEP}')

        # Execute the actions
        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_passes(actions)
        self._handle_drops()

        # Check for termination
        if self.check_termination():
            for agent in self.agents:
                agent.reward += REWARD_COMPLETION
                if self.debug_mode:
                    print(f'Rewarded for completion: {REWARD_COMPLETION}')
            self.done = True

        # Collect frames for the video when required
        if self.create_video:
            self.frames.append(
                pygame.surface.array3d(self.offscreen_surface))

        # Debug info
        if self.debug_mode:
            self._print_state()

        total_reward = sum(self._get_rewards())
        if self.debug_mode:
            print(f'Total reward: {total_reward}')
            for idx, agent in enumerate(self.agents):
                print(f"Agent {idx} Reward: {agent.reward}")

        obs = self.get_obs()

        return obs, total_reward, self.done, {}

    def _get_rewards(self) -> List[int]:
        return [agent.reward for agent in self.agents]

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

    # def _handle_moves(self, actions: List[int]) -> None:
    #     for idx, agent_action in enumerate(actions):
    #         agent = self.agents[idx]
    #         agent.position = self._move_agent(agent, agent_action)
    #
    #         # if an agent is carrying an object, the position of the object being carried needs to be updated
    #         if agent.carrying_object is not None:
    #             carried_object = next(
    #                 (o for o in self.objects if o.id == agent.carrying_object), None
    #             )
    #             if carried_object:
    #                 carried_object.position = agent.position

    def _handle_moves(self, actions: List[int]) -> None:
        for idx, agent_action in enumerate(actions):
            agent = self.agents[idx]
            new_position = self._move_agent(agent, agent_action)

            # Check if the agent is carrying an object and the new position has an object
            if agent.carrying_object is not None and any(obj.position == new_position for obj in self.objects):
                continue  # Skip the move if the agent is carrying an object and the new position has an object

            agent.position = new_position

            # If an agent is carrying an object, the position of the object being carried needs to be updated
            if agent.carrying_object is not None:
                carried_object = next(
                    (o for o in self.objects if o.id == agent.carrying_object), None)
                if carried_object:
                    carried_object.position = agent.position

    def _handle_pickups(self) -> None:
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying_object = obj.id
                        agent.reward += REWARD_PICKUP
                        if self.debug_mode:
                            print(f'Rewarded for pickup: {REWARD_PICKUP}')
                        break

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
                        agent.reward += REWARD_DROP
                        if self.debug_mode:
                            print(f'Rewarded for dropoff: {REWARD_DROP}')


    def _handle_passes(self, actions: List[int]) -> None:
        picker_to_non_picker_passes = []
        other_possible_passes = []

        for giver, giver_action in zip(self.agents, actions):
            if giver_action == Action.PASS.value and giver.carrying_object is not None:
                for adj_pos in self._get_adjacent_positions(giver.position):
                    receiver = self._find_agent_at_position(adj_pos)
                    if receiver:
                        receiver_action = actions[self.agents.index(receiver)]
                        if self._can_receive_object(giver, giver_action, receiver, receiver_action):
                            if giver.picker and not receiver.picker:
                                picker_to_non_picker_passes.append((giver, receiver, giver.carrying_object))
                            else:
                                other_possible_passes.append((giver, receiver, giver.carrying_object))

        if picker_to_non_picker_passes:
            giver, receiver, obj_id = random.choice(picker_to_non_picker_passes)
            giver.carrying_object = None
            receiver.carrying_object = obj_id
            self._reward_agents(giver, receiver)
        elif other_possible_passes:
            giver, receiver, obj_id = random.choice(other_possible_passes)
            giver.carrying_object = None
            receiver.carrying_object = obj_id
            self._reward_agents(giver, receiver)


    def _reward_agents(self, giver: Agent, receiver: Agent) -> None:
        if giver.reward is not None and receiver.reward is not None:
            if giver.picker and not receiver.picker:
                giver.reward += REWARD_GOOD_PASS
                receiver.reward += REWARD_GOOD_PASS
                self._log_reward("good", REWARD_GOOD_PASS)
            elif not giver.picker and receiver.picker:
                giver.reward += REWARD_BAD_PASS
                receiver.reward += REWARD_BAD_PASS
                self._log_reward("bad", REWARD_BAD_PASS)
        else:
            print("Warning: Attempted to update reward of agent with None reward.")


    def _find_agent_at_position(self, position: Tuple[int, int]) -> Optional[Agent]:
        return next((agent for agent in self.agents if agent.position == position), None)

    def _can_receive_object(self, giver: Agent, giver_action: int, receiver: Agent, receiver_action: int) -> bool:
        return (
            giver.carrying_object is not None and
            giver_action == Action.PASS.value and
            receiver_action == Action.PASS.value and
            receiver.carrying_object is None and
            not any(obj.position == receiver.position for obj in self.objects)
        )


    def _log_reward(self, pass_type: str, reward_amount: int) -> None:
        if self.debug_mode:
            print(f'Rewarded for {pass_type} pass: {reward_amount}')

    def _get_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def check_termination(self) -> bool:
        # Check if all objects are on their goal positions and not being carried
        for obj in self.objects:
            if obj.position not in self.goals or any(agent.carrying_object == obj.id for agent in self.agents):
                return False

        # Check if no non-picker agent is carrying an object
        for agent in self.agents:
            if not agent.picker and agent.carrying_object is not None:
                return False

        return True

    def _init_render(self) -> None:
        from macpp.core.rendering import Viewer
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
