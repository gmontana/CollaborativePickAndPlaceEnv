import random
import pygame

ANIMATION_DELAY = 400
REWARD_STEP = -1
REWARD_PASS = 1
REWARD_GOAL = 2
REWARD_COMPLETION = 20


class Agent:
    def __init__(self, position, picker, carrying_object=None):
        self.position = position
        self.picker = picker
        self.carrying_object = carrying_object
        self.reward = 0


class Object:
    def __init__(self, position, obj_id):
        self.position = position
        self.id = obj_id


class MultiAgentPickAndPlace:
    def __init__(
        self,
        width,
        length,
        n_agents,
        n_pickers,
        initial_state=None,
        cell_size=100,
        enable_printing=False,
        enable_rendering=False,
    ):
        self.width = width
        self.length = length
        self.cell_size = cell_size
        self.n_agents = n_agents
        self.enable_printing = enable_printing
        self.enable_rendering = enable_rendering
        self.n_pickers = n_pickers
        self.objects = []
        self.goals = []
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
            pygame.display.set_caption("Multi-Agent Pick and Place")

            # self.agent_icons = [
            #     pygame.image.load("icons/agent1.png"), 
            #     pygame.image.load("icons/agent2.png"),
            # ]

            

    def _validate_actions(self, actions):
        for action in actions:
            if action not in self.action_space:
                raise ValueError(f"Unrecognized action: {action}.")

    def random_initialize(self):
        all_positions = [(x, y) for x in range(self.width) for y in range(self.length)]
        random.shuffle(all_positions)

         # Initialize objects with distinct positions
        self.objects = [Object(all_positions.pop(), obj_id=i) for i in range(self.n_agents)]

        # Initialize agents with distinct positions
        self.agents = [Agent(position=all_positions.pop(), picker=picker_flags[i]) for i in range(self.n_agents)]

        # Assign goals with distinct positions
        self.goals = [all_positions.pop() for _ in range(self.n_agents)]

    def initialize_from_state(self, initial_state):
        # Initialise objects
        self.objects = [
            Object(position=obj["position"], obj_id=obj.get("id", None))
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

        rewards = [REWARD_STEP] * self.n_agents

        self._handle_moves(actions)
        self._handle_drops()
        self._handle_pickups()
        self._handle_pass_actions(actions)
        termination_reward = self._check_termination()

        if termination_reward:
            for idx in range(self.n_agents):
                rewards[idx] += termination_reward

        if self.enable_printing:
            self.print_state()

        return rewards

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
        # Only check for collisions with other agents
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
            print(f"Before moving: Agent at {agent.position} is carrying: {agent.carrying_object}")
            if agent.picker and agent.carrying_object is None:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying_object = obj.id  # Using the object's id
                        objects_to_remove.append(obj)
                        # print(
                        #     f"Agent {self.agents.index(agent) + 1} picked up object with ID {obj.id} at position {obj.position}"
                        # )
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
        agent_positions_with_objects = [
            (agent.position, agent.carrying_object)
            for agent in self.agents
            if agent.carrying_object is not None
        ]

        if all(
            pos in goal_positions for pos, obj_id in agent_positions_with_objects
        ) and len(set(obj_id for _, obj_id in agent_positions_with_objects)) == len(
            agent_positions_with_objects
        ):
            self.done = True
            return REWARD_COMPLETION
        return 0

    def _handle_drops(self):
        for agent in self.agents:
            if agent.carrying_object is not None:
                print(f"Agent at {agent.position} is carrying object {agent.carrying_object}")
                
                if not agent.picker:
                    print(f"Agent at {agent.position} is not a picker")
                    
                    if agent.position in self.goals:
                        print(f"Agent at position {agent.position} is trying to drop object {agent.carrying_object}")
                        
                        obj = next((o for o in self.objects if o.id == agent.carrying_object), None)
                        if obj:
                            obj.position = agent.position
                            print(f"Object {obj.id} dropped at {obj.position}")
                        else:
                            print(f"No object found with ID {agent.carrying_object}")
                        
                        agent.carrying_object = None
                        agent.reward += REWARD_GOAL
                    else:
                        print(f"Agent at {agent.position} isn't on a goal")
                else:
                    print(f"Agent at {agent.position} is a picker")
            else:
                print(f"Agent at {agent.position} isn't carrying an object")



    def render(self):

        # Define colors
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLACK = (0, 0, 0)
        GRAY = (200, 200, 200)
        YELLOW = (255, 255, 0)

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

        # # Draw agents with icon images
        # for idx, agent in enumerate(self.agents):
        #     x, y = agent.position
        #     agent_icon = self.agent_icons[idx] if idx < len(self.agent_icons) else None
        #     if agent_icon:
        #         cell_center = (
        #             x * self.cell_size + self.cell_size // 2,
        #             y * self.cell_size + self.cell_size // 2
        #         )
        #         scaling_factor = 0.8 
        #         icon_size = int(self.cell_size * scaling_factor)
        #         agent_icon_resized = pygame.transform.scale(agent_icon, (icon_size, icon_size))
        #         agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
        #         self.screen.blit(agent_icon_resized, agent_icon_rect)
       
        # Draw agents
        for agent in self.agents:
            x, y = agent.position
            color = RED if agent.picker else BLUE
            if agent.carrying_object is not None:
                # Draw an additional shape or symbol to represent carrying agents
                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2,
                    ),
                    self.cell_size // 3,
                )
                # Draw a smaller rectangle inside the agent's cell to represent the object they're carrying
                pygame.draw.rect(
                    self.screen,
                    YELLOW,
                    (
                        x * self.cell_size + self.cell_size // 3,
                        y * self.cell_size + self.cell_size // 3,
                        self.cell_size // 3,
                        self.cell_size // 3,
                    ),
                )
            else:
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * self.cell_size + self.cell_size // 4,
                        y * self.cell_size + self.cell_size // 4,
                        self.cell_size // 2,
                        self.cell_size // 2,
                    ),
                )

        # Draw objects (use smaller circles to represent objects)
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
                GRAY,
                (
                    x * self.cell_size + self.cell_size // 3,
                    y * self.cell_size + self.cell_size // 3,
                    self.cell_size // 3,
                    self.cell_size // 3,
                ),
            )

        pygame.display.flip()

        # We also add a delay to slow down the rendering speed
        pygame.time.wait(ANIMATION_DELAY)

        # Event loop to close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # If the game is over, we can add additional logic to close the window or restart.
        if self.done:
            pygame.quit()
