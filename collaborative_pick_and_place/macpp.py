import random
import pygame
import json 

ANIMATION_DELAY = 150
REWARD_STEP = -1
REWARD_PASS = 1
REWARD_GOAL = 5
REWARD_COMPLETION = 20

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

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
            "reward": self.reward
        }


class Object:
    def __init__(self, position, id):
        self.position = position
        self.id = id
    def get_state(self):
            return {
                "id": self.id,
                "position": self.position
            }

class MultiAgentPickAndPlace:
    def __init__(
        self,
        width,
        length,
        n_agents,
        n_pickers,
        initial_state=None,
        cell_size=100,
        debug_mode=False,
        enable_rendering=False,
    ):
        self.width = width
        self.length = length
        self.cell_size = cell_size
        self.n_agents = n_agents
        self.debug_mode = debug_mode
        self.enable_rendering = enable_rendering
        self.n_pickers = n_pickers
        self.objects = []
        self.goals = []
        self.initial_state = initial_state
        if initial_state is None:
            self.random_initialize()
        else:
            self.initialize_from_state(initial_state)

        self.action_space = ["move_up", "move_down", "move_left", "move_right", "pass"]
        self.done = False

        if self.enable_rendering:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.length * self.cell_size)
            )
            pygame.display.set_caption("Collaborative Multi-Agent Pick and Place")
            picker_icon = pygame.image.load("icons/agent_picker.png")
            non_picker_icon = pygame.image.load("icons/agent_non_picker.png")
            self.agent_icons = [picker_icon if agent.picker else non_picker_icon for agent in self.agents]


    def _validate_actions(self, actions):
        for action in actions:
            if action not in self.action_space:
                raise ValueError(f"Unrecognized action: {action}.")


    def reset(self):
        '''
        Reset the environment to either a random state or an predefined initial state
        '''
        if hasattr(self, 'initial_state') and self.initial_state is not None:
            self.initialize_from_state(self.initial_state)
        else:
            self.random_initialize()

        for agent in self.agents:
            agent.reward = 0
            agent.carrying_object = None
        self.done = False

    def get_hashed_state(self):
        '''
        Return the hashed current state
        '''
        agent_states = tuple(json.dumps(agent.get_state()) for agent in self.agents)
        object_states = tuple(json.dumps(obj.get_state()) for obj in self.objects)
        goals = tuple(self.goals)
        combined_state = agent_states + object_states + goals
        return hash(combined_state)



    def get_action_space(self):
        '''
        Return the action space of the environment
        '''
        return self.action_space


    def random_initialize(self):
        '''
        Initialise the environment in a random state
        '''
        all_positions = [(x, y) for x in range(self.width) for y in range(self.length)]
        random.shuffle(all_positions)

        # Initialize objects with distinct positions
        self.objects = [Object(all_positions.pop(), id=i) for i in range(self.n_agents)]

        # Create a list of picker flags based on the number of pickers
        picker_flags = [True] * self.n_pickers + [False] * (self.n_agents - self.n_pickers)
        random.shuffle(picker_flags)  # Shuffle to randomize which agents are pickers

        # Initialize agents with distinct positions
        self.agents = [Agent(position=all_positions.pop(), picker=picker) for picker in picker_flags]

        # Assign goals with distinct positions
        self.goals = [all_positions.pop() for _ in range(self.n_agents)]


    def initialize_from_state(self, initial_state):
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
                position=(agent_x, agent_y), picker=picker, carrying_object=carrying_object
            )
            self.agents.append(agent)

    def print_state(self):
        print("="*40)
        print("Agents' State:")
        for idx, agent in enumerate(self.agents, start=1):
            carrying_status = "Carrying" if agent.carrying_object is not None else "Not Carrying"
            carrying_object = (
                f"Object ID: {agent.carrying_object}" if agent.carrying_object is not None else "None"
            )
            print(
                f"- Agent {idx:<2}: Position: {agent.position}, "
                f"Picker: {str(agent.picker):<5}, Status: {carrying_status:<12}, {carrying_object}"
            )

        print("\nObjects' Positions:")
        for idx, obj in enumerate(self.objects, start=1):
            print(f"- Object {idx}: Position: {obj.position}, ID: {obj.id}")  # <-- This line is changed

        print("Goal Positions:")
        if not self.goals:
            print("- No goal positions set.")
        else:
            for idx, goal in enumerate(self.goals):
                print(f"- Goal {idx + 1}: Position {goal}")
    

            
    def _random_position(self):
        return (random.randint(0, self.width - 1), random.randint(0, self.length - 1))





    def step(self, actions):

        self._validate_actions(actions)

        # Negative reward given at every step 
        rewards = [REWARD_STEP] * self.n_agents

        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_pass_actions(actions)
        termination_reward = self._check_termination()

        done = False
        if termination_reward:
            for idx in range(self.n_agents):
                rewards[idx] += termination_reward
            done = True

        next_state = {
            "agents": [agent.get_state() for agent in self.agents],
            "objects": [obj.get_state() for obj in self.objects],
            "goals": self.goals
        }

        if self.debug_mode:
            self.print_state()

        if self.enable_rendering:
            self.render()

        if self.debug_mode:
            self._check_state_integrity(actions)

        return next_state, rewards, done

    def _check_state_integrity(self, actions):
        agent_positions = [agent.position for agent in self.agents]
        object_positions = [obj.position for obj in self.objects]

        assert len(agent_positions) == len(set(agent_positions)), "Two agents have the same position!"

        for agent in self.agents:
            assert agent.position not in object_positions or agent.carrying_object is not None, "Agent and object overlap without pickup!"

        for agent in self.agents:
            if agent.carrying_object is not None:
                assert not any(obj.id == agent.carrying_object for obj in self.objects), "Carried object still on grid!"

        assert len(self.goals) == len(self.agents), "Mismatch between number of agents and goals!"

        assert len(object_positions) == len(set(object_positions)), "Two objects have the same position!"

        for obj in self.objects:
            assert obj.position not in self.goals, "Object and goal overlap without placement!"

        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                assert agent.position not in self.goals, "Picker agent dropped an object!"

        for agent in self.agents:
            if not agent.picker and agent.carrying_object is not None:
                adjacent_positions = [
                    (agent.position[0] + dx, agent.position[1] + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ]
                assert any(
                    other_agent.position in adjacent_positions and other_agent.carrying_object is None
                    for other_agent in self.agents
                ), "Non-picker agent picked an object!"

        for idx, action in enumerate(actions):
            if action == "pass" and self.agents[idx].carrying_object is None:
                adjacent_positions = [
                    (self.agents[idx].position[0] + dx, self.agents[idx].position[1] + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ]
                assert not any(
                    other_agent.position in adjacent_positions
                    for other_agent in self.agents
                ), "Agent not carrying an object tried to pass!"

        for idx, action in enumerate(actions):
            if action == "pass" and self.agents[idx].carrying_object is not None:
                adjacent_positions = [
                    (self.agents[idx].position[0] + dx, self.agents[idx].position[1] + dy)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ]
                assert not any(
                    other_agent.position in adjacent_positions and other_agent.carrying_object is not None
                    for other_agent in self.agents
                ), "Agent tried to pass an object to another agent already carrying an object!"



    def _move_agent(self, agent, action):

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

    def _handle_pickups(self):
        objects_to_remove = []
        for agent in self.agents:
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying_object = obj.id  
                        objects_to_remove.append(obj)
                        break
        for obj in objects_to_remove:
            self.objects.remove(obj)

    def _handle_pass_actions(self, actions):

        # All agents whose action is a 'pass'
        passing_agents = [
            agent for idx, agent in enumerate(self.agents) if actions[idx] == "pass"
        ]

        # Filter carrying agents from those who want to pass
        carrying_agents = [agent for agent in passing_agents if agent.carrying_object is not None]

        for agent in carrying_agents:
            x, y = agent.position
            adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            
            # Filter eligible agents from those passing adjacent to the carrying agent
            adjacent_passing_agents = [
                adj_agent
                for adj_agent in passing_agents
                if adj_agent.position in adjacent_positions and adj_agent.carrying_object is None
            ]

            if adjacent_passing_agents:
                chosen_adj_agent = random.choice(adjacent_passing_agents)
                
                # Give rewards only when object is passed from Picker to non-Picker
                agent.reward += REWARD_PASS
                chosen_adj_agent.reward += REWARD_PASS
                
                chosen_adj_agent.carrying_object = agent.carrying_object
                agent.carrying_object = None

    def _check_termination(self):
        goal_positions = set(self.goals)
        object_positions = [obj.position for obj in self.objects]

        # Check if all object positions are in goal positions
        if all(pos in goal_positions for pos in object_positions):
            self.done = True
            return REWARD_COMPLETION
        return 0



    def _handle_drops(self):
        for agent in self.agents:
            if agent.carrying_object is not None:
                if not agent.picker:
                    if agent.position in self.goals:
                        obj = next((o for o in self.objects if o.id == agent.carrying_object), None)
                        if obj:
                            obj.position = agent.position
                        agent.carrying_object = None
                        agent.reward += REWARD_GOAL


    def render(self):


        # Fill background
        self.screen.fill(WHITE)

        # Draw grid
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.screen, BLACK, (x, 0), (x, self.length * self.cell_size)
            )
        for y in range(0, self.length * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.screen, BLACK, (0, y), (self.width * self.cell_size, y)
            )

        # Draw objects
        for obj in self.objects:
            x, y = obj.position
            pygame.draw.circle(
                self.screen,
                GREEN,
                (
                    x * self.cell_size + self.cell_size // 2,
                    y * self.cell_size + self.cell_size // 2,
                ),
                self.cell_size // 4,
            )

        # Draw goals (small rectangles)
        for goal in self.goals:
            x, y = goal
            pygame.draw.rect(
                self.screen,
                LIGHT_GRAY,
                (
                    x * self.cell_size + self.cell_size // 3,
                    y * self.cell_size + self.cell_size // 3,
                    self.cell_size // 3,
                    self.cell_size // 3,
                ),
            )

        # Draw agents with icon images
        for idx, agent in enumerate(self.agents):
            x, y = agent.position
            agent_icon = self.agent_icons[idx] if idx < len(self.agent_icons) else None
            if agent_icon:
                cell_center = (
                    x * self.cell_size + self.cell_size // 2,
                    y * self.cell_size + self.cell_size // 2
                )
                scaling_factor = 0.8 
                icon_size = int(self.cell_size * scaling_factor)
                agent_icon_resized = pygame.transform.scale(agent_icon, (icon_size, icon_size))
                agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
                self.screen.blit(agent_icon_resized, agent_icon_rect)

                # Agent is carrying an object 
                if agent.carrying_object is not None:
                    thickness = 3
                    pygame.draw.rect(self.screen, GREEN, agent_icon_rect, thickness)


        # # Draw agents
        # for agent in self.agents:
        #     x, y = agent.position
        #     color = RED if agent.picker else BLUE
        #     if agent.carrying_object is not None:
        #         # Draw an additional shape or symbol to represent carrying agents
        #         pygame.draw.circle(
        #             self.screen,
        #             color,
        #             (
        #                 x * self.cell_size + self.cell_size // 2,
        #                 y * self.cell_size + self.cell_size // 2,
        #             ),
        #             self.cell_size // 3,
        #         )
        #         # Draw a smaller rectangle inside the agent's cell to represent the object they're carrying
        #         pygame.draw.rect(
        #             self.screen,
        #             YELLOW,
        #             (
        #                 x * self.cell_size + self.cell_size // 3,
        #                 y * self.cell_size + self.cell_size // 3,
        #                 self.cell_size // 3,
        #                 self.cell_size // 3,
        #             ),
        #         )
        #     else:
        #         pygame.draw.rect(
        #             self.screen,
        #             color,
        #             (
        #                 x * self.cell_size + self.cell_size // 4,
        #                 y * self.cell_size + self.cell_size // 4,
        #                 self.cell_size // 2,
        #                 self.cell_size // 2,
        #             ),
        #         )

        pygame.display.flip()
        pygame.time.wait(ANIMATION_DELAY)

