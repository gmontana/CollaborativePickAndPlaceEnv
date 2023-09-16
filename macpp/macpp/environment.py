import gym
# from enum import Enum
from gym import spaces
import random
import pygame
import json
import imageio
import os
import numpy as np
import time
from rendering import Viewer
# import hashlib

REWARD_STEP = -1
REWARD_GOOD_PASS = 5
REWARD_BAD_PASS = -5
REWARD_DROP = 10
REWARD_COMPLETION = 20

# class Action(Enum):
#     MOVE_UP = 1
#     MOVE_DOWN = 2
#     MOVE_LEFT = 3
#     MOVE_RIGHT = 4
#     PASS = 5
    
class Agent:
    def __init__(self, position, picker, carrying_object=None):
        self.position = position
        self.picker = picker
        self.carrying_object = carrying_object
        self.reward = 0

    def get_state(self):
        return {
            "position": self.position,
            "picker": self.picker,
            "carrying_object": self.carrying_object,
            "reward": self.reward,
        }


class Object:
    def __init__(self, position, id):
        self.position = position
        self.id = id

    def get_state(self):
        return {"id": self.id, "position": self.position}


class MultiAgentPickAndPlace(gym.Env):
    """ 
    Class containing the logic for the collaborating pick and place environment
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        width,
        length,
        n_agents,
        n_pickers,
        n_objects=None,  # default to number of agents
        initial_state=None,
        cell_size=100,
        debug_mode=False,
        enable_rendering=False,
        create_video=False,
    ):
        self.width = width
        self.length = length
        self.cell_size = cell_size
        self.n_agents = n_agents
        self.debug_mode = debug_mode
        self.enable_rendering = enable_rendering
        self.n_pickers = n_pickers
        self.agents = []
        self.objects = []
        self.goals = []
        self.initial_state = initial_state
        self.create_video = create_video

        # Define actions and done flag
        self.action_space = ["move_up", "move_down", "move_left", "move_right", "pass"]
        self.action_space = spaces.Discrete(len(self.action_space))
        self.observation_space = spaces.Discrete(1e6)
        self.done = False

        # Set the number of objects and goals
        if n_objects is None:
            self.n_objects = self.n_agents
        else:
            self.n_objects = n_objects

        # Use a random state unless a predefined state is provided
        if initial_state is None:
            self.random_initialize()
        else:
            self.initialize_from_state(initial_state)

        # Check whether the grid size is sufficiently large
        total_cells = self.width * self.length
        total_entities = self.n_agents + self.n_objects
        if total_entities > total_cells:
            raise ValueError(
                "Grid size not sufficiently large to contain all the entities."
            )

        # Initialise pygame
        pygame.init()

        # Create the offscreen surface for rendering
        self.offscreen_surface = pygame.Surface(
            (self.width * self.cell_size, self.length * self.cell_size)
        )

        # Load agent icons
        image_path = os.path.join(
            os.path.dirname(__file__), "icons", "agent_picker.png"
        )
        self.picker_icon = pygame.image.load(image_path)

        # When rendering is required, create the screen for display
        if self.enable_rendering:
            self.viewer = Viewer(width, length, cell_size)
            # self.screen = pygame.display.set_mode(
            #     (self.width * self.cell_size, self.length * self.cell_size)
            # )
            # pygame.display.set_caption("Collaborative Multi-Agent Pick and Place")

        # If a video is required, create frames 
        if self.create_video:
            self.frames = []


    def _validate_actions(self, actions):
        for action in actions:
            if action not in self.get_action_space():
                raise ValueError(f"Unrecognized action: {action}.")

    def reset(self):
        """
        Reset the environment to either a random state or an predefined initial state
        """
        if hasattr(self, "initial_state") and self.initial_state is not None:
            self.initialize_from_state(self.initial_state)
        else:
            self.random_initialize()

        for agent in self.agents:
            agent.reward = 0
            agent.carrying_object = None
        self.done = False


        # agent_states = [agent.get_state() for agent in self.agents]
        # object_states = [obj.get_state() for obj in self.objects]
        # goal_states = self.goals

        return self.get_hashed_state()


    def get_hashed_state(self):
        """
        Return the hashed current state
        """
        agent_states = tuple((agent.position, agent.picker, agent.carrying_object) for agent in self.agents)
        object_states = tuple(obj.position for obj in self.objects)
        goals = tuple(self.goals)
        combined_state = agent_states + object_states + goals
        return hash(combined_state)

    def get_action_space(self):
        """
        Return the action space of the environment
        """
        return self.action_space

    def random_initialize(self):
        """
        Initialise the environment with random allocations of agents and objects
        """
        all_positions = [(x, y) for x in range(self.width) for y in range(self.length)]
        random.shuffle(all_positions)

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
                )
            )

        # Initialize objects
        self.objects = [
            Object(position=obj_pos, id=i) for i, obj_pos in enumerate(object_positions)
        ]

        # Assign goals
        self.goals = goal_positions

    def initialize_from_state(self, initial_state):
        """
        Initiate environment at a predefined state
        """

        # Initialise objects
        self.objects = [
            Object(position=obj["position"], id=obj.get("id", None))
            for obj in initial_state["objects"]
        ]

        # Initialise goals
        self.goals = initial_state.get("goals", [])

        # Initialise agents
        self.agents = []
        for i in range(self.n_agents):
            agent_x, agent_y = initial_state["agents"][i]["position"]
            picker = initial_state["agents"][i]["picker"]

            # Assign the carrying_object from the initial state if it exists.
            carrying_object = initial_state["agents"][i].get("carrying_object", None)

            agent = Agent(
                position=(agent_x, agent_y),
                picker=picker,
                carrying_object=carrying_object,
            )
            self.agents.append(agent)

    def _print_state(self):
        print("=" * 40)
        print("Agents' State:")
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

        print("\nObjects' Positions:")
        for idx, obj in enumerate(self.objects, start=1):
            print(f"- Object {idx}: Position: {obj.position}, ID: {obj.id}")

        print("Goal Positions:")
        if not self.goals:
            print("- No goal positions set.")
        else:
            for idx, goal in enumerate(self.goals):
                print(f"- Goal {idx + 1}: Position {goal}")

    def _random_position(self):
        return (random.randint(0, self.width - 1), random.randint(0, self.length - 1))

    def step(self, actions):
        """
        Step through the environment 
        """

        self._validate_actions(actions)

        # Negative reward given at every step
        rewards = [REWARD_STEP] * self.n_agents

        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_passes(actions)
        termination_reward = self.check_termination()

        done = False
        if termination_reward:
            for idx in range(self.n_agents):
                rewards[idx] += termination_reward
            done = True

        next_state = {
            "agents": [agent.get_state() for agent in self.agents],
            "objects": [obj.get_state() for obj in self.objects],
            "goals": self.goals,
        }

        if self.debug_mode:
            self._print_state()

        if self.enable_rendering:
            self.render()

        # if self.create_video:
        #     self.frames = []
        # Collect frames for the video when required and draw
        if self.create_video:
            self.frames.append(pygame.surfarray.array3d(self.offscreen_surface))

        next_state_hash = self.get_hashed_state()
        return next_state_hash, rewards, done, {}

    def _move_agent(self, agent, action):
        """
        Move the envirnment in the grid. Collisions between agents are not allowed.
        """

        x, y = agent.position
        if action == "move_up":
            y = max(0, y - 1)
        elif action == "move_down":
            y = min(self.length - 1, y + 1)
        elif action == "move_left":
            x = max(0, x - 1)
        elif action == "move_right":
            x = min(self.width - 1, x + 1)

        new_position = (x, y)
        # Check for collisions with other agents
        if new_position not in [a.position for a in self.agents]:
            return new_position
        return agent.position

    def _handle_moves(self, actions):
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

    def _handle_pickups(self):
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying_object = obj.id
                        break

    def _handle_passes(self, actions):

        # Create a list to store agents that will receive objects
        receiving_agents = [None] * len(self.agents)

        for idx, agent in enumerate(self.agents):
            if actions[idx] == "pass" and agent.carrying_object is not None:
                x, y = agent.position
                adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for adj_pos in adjacent_positions:
                    adj_agent = next(
                        (a for a in self.agents if a.position == adj_pos), None
                    )
                    if (
                        adj_agent
                        and actions[self.agents.index(adj_agent)] == "pass"
                        and adj_agent.carrying_object is None
                    ):
                        receiving_agents[
                            self.agents.index(adj_agent)
                        ] = agent.carrying_object
                        obj = next(
                            (o for o in self.objects if o.id == agent.carrying_object),
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

    def check_termination(self):
        goal_positions = set(self.goals)
        object_positions = [obj.position for obj in self.objects]

        for obj in self.objects:
            if obj.position in goal_positions:
                agent = next(
                    (
                        a
                        for a in self.agents
                        if a.carrying_object == obj.id and not a.picker
                    ),
                    None,
                )
                if agent:
                    return REWARD_COMPLETION

        return 0

    def _handle_drops(self):
        for agent in self.agents:
            if agent.carrying_object is not None:
                if not agent.picker:
                    if agent.position in self.goals:
                        obj = next(
                            (o for o in self.objects if o.id == agent.carrying_object),
                            None,
                        )
                        if obj:
                            obj.position = agent.position
                        else:
                            # Add the object back to the environment's list of objects
                            new_obj = Object(
                                position=agent.position, id=agent.carrying_object
                            )
                            self.objects.append(new_obj)
                        agent.carrying_object = None
                        agent.reward += REWARD_DROP

    def render(self):
        if self.viewer:
            self.viewer.render(self.agents, self.objects, self.goals)

    def close(self):
        if self.viewer:
            self.viewer.close()

def _game_loop(env, render):
    """
    Game loop for the MultiAgentPickAndPlace environment.
    """
    obs = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        # Sample random actions for each agent
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        print(actions)

        nobs, nreward, ndone, _ = env.step(actions)
        print(nreward)
        # if sum(nreward) > 0:
        #     print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)

if __name__ == "__main__":
    env = MultiAgentPickAndPlace(
        width=5,
        length=5,
        n_agents=2,
        n_pickers=1,
        cell_size=3,
        enable_rendering=True)
    for episode in range(10):
        print(f"Episode {episode}:")
        _game_loop(env,  render = True)
    print("Done")

