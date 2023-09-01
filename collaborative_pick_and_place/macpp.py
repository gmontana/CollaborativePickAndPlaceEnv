import random
import pygame

ANIMATION_DELAY = 400
REWARD_STEP = -1
REWARD_PASS = 1
REWARD_GOAL = 2
REWARD_COMPLETION = 20


class Agent:
    def __init__(self, position, picker, carrying=False, carrying_object=None):
        self.position = position
        self.picker = picker
        self.carrying = carrying
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

    def _validate_actions(self, actions):
        for action in actions:
            if action not in self.action_space:
                raise ValueError(f"Unrecognized action: {action}.")

    def random_initialize(self):
        all_positions = [(x, y) for x in range(self.width) for y in range(self.length)]
        random.shuffle(all_positions)

        picker_flags = [True] * self.n_agents
        random.shuffle(picker_flags)
        for i in range(self.n_pickers, self.n_agents):
            picker_flags[i] = False

        self.objects = [
            Object(all_positions.pop(), obj_id=i) for i in range(self.n_agents)
        ]

        # Initialize agents with distinct positions
        self.agents = [
            Agent(position=all_positions.pop(), picker=picker_flags[i])
            for i in range(self.n_agents)
        ]

        # Assign goals with distinct positions
        self.goals = [all_positions.pop() for _ in range(self.n_agents)]

    def initialize_from_state(self, initial_state):
        # First, initialize objects
        self.objects = [
            Object(position=obj["position"], obj_id=obj.get("id", None))
            for obj in initial_state["objects"]
        ]

        # Then, initialize agents
        self.agents = []
        for i in range(self.n_agents):
            agent_x, agent_y = initial_state["agents"][i]["position"]
            picker = initial_state["agents"][i]["picker"]

            # Check if "id" key exists for the object; if not, assign None
            object_id = None
            if i < len(
                initial_state["objects"]
            ):  # Ensure we're not going out of bounds
                object_id = initial_state["objects"][i].get("id", None)

            agent = Agent(
                position=(agent_x, agent_y), picker=picker, carrying_object=object_id
            )
            self.agents.append(agent)

    def print_state(self):
        print("Agents' Positions:")
        for idx, agent in enumerate(self.agents):
            print(f"- Agent {idx + 1}: Position {agent.position}")

        print("Agents' Attributes:")
        for idx, agent in enumerate(self.agents):
            carrying_status = "Carrying" if agent.carrying else "Not Carrying"
            carrying_object = (
                f", Carrying Object ID: {agent.carrying_object}"
                if agent.carrying_object is not None
                else ""
            )
            print(
                f"- Agent {idx + 1}: Picker: {agent.picker}, {carrying_status}{carrying_object}"
            )

        print("Goal Positions:")
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
            if agent.picker and not agent.carrying:
                for obj in self.objects:
                    if obj.position == agent.position:
                        agent.carrying = True
                        agent.carrying_object = obj.id  # Using the object's id
                        objects_to_remove.append(obj)
                        # print(
                        #     f"Agent {self.agents.index(agent) + 1} picked up object with ID {obj.id} at position {obj.position}"
                        # )
                        break
        for obj in objects_to_remove:
            self.objects.remove(obj)

    def _handle_pass_actions(self, actions):
        passing_agents = [
            agent for idx, agent in enumerate(self.agents) if actions[idx] == "pass"
        ]
        carrying_agents = [agent for agent in passing_agents if agent.carrying]

        for agent in carrying_agents:
            x, y = agent.position
            adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            adjacent_passing_agents = [
                adj_agent
                for adj_agent in passing_agents
                if adj_agent.position in adjacent_positions and not adj_agent.carrying
            ]

            if adjacent_passing_agents:
                chosen_adj_agent = random.choice(adjacent_passing_agents)
                agent.reward += REWARD_PASS
                chosen_adj_agent.reward += REWARD_PASS
                agent.carrying = False
                chosen_adj_agent.carrying = True
                chosen_adj_agent.carrying_object = agent.carrying_object
                agent.carrying_object = None

    def _check_termination(self):
        goal_positions = set(self.goals)
        agent_positions_with_objects = [
            (agent.position, agent.carrying_object)
            for agent in self.agents
            if agent.carrying
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
            if agent.carrying and not agent.picker and agent.position in self.goals:
                if agent.carrying_object is not None:
                    for obj in self.objects:
                        if obj.id == agent.carrying_object:
                            obj.position = agent.position
                            break
                    agent.carrying_object = None
                    agent.carrying = False
                    agent.reward += REWARD_GOAL
                else:
                    print(
                        f"Warning: Agent {self.agents.index(agent) + 1} is at goal position {agent.position} and is carrying, but carrying_object is None."
                    )

    def get_carrying_object_position(self, agent):
        if agent.carrying_object is not None:
            obj_id = agent.carrying_object
            for obj in self.objects:
                if obj.id == obj_id:
                    return obj.position
        return None

    def set_carrying_object_position(self, agent, new_position):
        if agent.carrying_object is not None:
            obj_id = agent.carrying_object
            for obj in self.objects:
                if obj.id == obj_id:
                    obj.position = new_position
                    break

    def reset(self):
        all_positions = [(x, y) for x in range(self.width) for y in range(self.length)]
        random.shuffle(all_positions)

        self.objects = [Object(all_positions.pop()) for _ in range(self.n_agents)]
        self.agents = [
            Agent(all_positions.pop(), picker=random.choice([True, False]))
            for _ in range(self.n_agents)
        ]
        self.goals = [all_positions.pop() for _ in range(self.n_agents)]
        self.done = False

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

        # Draw agents
        for agent in self.agents:
            x, y = agent.position
            color = RED if agent.picker else BLUE
            if agent.carrying:
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
