from macpp import MultiAgentPickAndPlace
import unittest


class TestPassingLogic(unittest.TestCase):
    def test_invalid_moves(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying": False},
                {"position": (1, 0), "picker": False, "carrying": False},
            ],
            "objects": [
                {"position": (2, 2), "id": 0},
            ],
            "goals": [
                (0, 0),
                (1, 0),
            ],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 0, initial_state=initial_state)
        actions = [
            "move_right",
            "move_left",
        ]
        env.step(actions)
        self.assertEqual(
            env.agents[0].position,
            (0, 0),
            "Agent moved out of bounds or overlapped with another agent.",
        )

    def test_dropoff_object(self):
        initial_state = {
            "agents": [{"position": (2, 2), "picker": False, "carrying": True}],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(1, 2)],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state)
        actions = ["move_left"]
        env.step(actions)
        self.assertFalse(env.agents[0].carrying, "Agent failed to drop off the object.")

    def test_termination_condition(self):
        initial_state = {
            "agents": [
                {"position": (3, 3), "picker": False, "carrying": True},
                {"position": (4, 4), "picker": False, "carrying": True},
                {"position": (5, 5), "picker": False, "carrying": True},
                {"position": (6, 6), "picker": False, "carrying": True},
            ],
            "objects": [
                {"position": (3, 3), "id": 0},
                {"position": (4, 4), "id": 1},
                {"position": (5, 5), "id": 2},
                {"position": (6, 6), "id": 3},
            ],
            "goals": [
                (3, 4),
                (5, 4),
                (6, 5),
                (7, 6),
            ],
        }
        env = MultiAgentPickAndPlace(10, 10, 4, 4, initial_state=initial_state)
        actions = ["move_down", "move_right", "move_right", "move_right"]
        env.step(actions)
        self.assertTrue(env.done, "Termination condition not recognized.")

    def test_pass_object_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying": True},
                {"position": (1, 0), "picker": True, "carrying": False},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 1, initial_state=initial_state)
        actions = ["pass", "pass"]
        env.step(actions)
        self.assertFalse(env.agents[0].carrying, "Agent 1 did not pass the object")
        self.assertTrue(env.agents[1].carrying, "Agent 2 did not receive the object")

    def test_multiple_agents_pick_same_object(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying": False},
                {"position": (0, 1), "picker": True, "carrying": False},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 2, initial_state=initial_state)
        actions = ["pass", "move_left"]
        env.step(actions)
        # Only one agent should be able to pick up the object
        self.assertTrue(
            env.agents[0].carrying or env.agents[1].carrying,
            "Multiple agents picked up the same object.",
        )
        self.assertNotEqual(
            env.agents[0].carrying,
            env.agents[1].carrying,
            "Both agents managed to pick up the object.",
        )

    def test_multiple_agent_passes(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying": True},
                {"position": (1, 0), "picker": True, "carrying": True},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (1, 0), "id": 1}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 2, initial_state=initial_state)
        actions = ["pass", "pass"]
        env.step(actions)
        self.assertTrue(
            env.agents[0].carrying and env.agents[1].carrying,
            "Agents failed to handle multiple passes.",
        )

    def test_invalid_pass(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": True, "carrying": True}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state)
        actions = ["pass"]
        env.step(actions)
        self.assertTrue(
            env.agents[0].carrying, "Agent passed the object when it shouldn't have."
        )

    def test_non_picker_cannot_pick(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": False, "carrying": False}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state)
        actions = ["pass"]
        env.step(actions)
        self.assertFalse(
            env.agents[0].carrying,
            "Agent that's not a picker managed to pick an object.",
        )


if __name__ == "__main__":

    # unittest.main()

    initial_state = {
        "agents": [
            {"position": (1, 1), "picker": True, "carrying_object": 0},
            {"position": (3, 3), "picker": False, "carrying_object": None},
            {"position": (4, 4), "picker": True, "carrying_object": None},
        ],
        "objects": [{"position": (1, 0), "id": 0}],
        "goals": [(0, 3), (3, 3), (0, 2)],
    }

    environment = MultiAgentPickAndPlace(
        width=5,
        length=5,
        n_agents=3,
        n_pickers=2,
        initial_state=initial_state,
        enable_rendering=True,
    )

    print(environment.print_state())

    time_steps = [
        ["move_right", "move_up", "move_left"],  # Agents move
        ["move_down", "move_left", "move_left"],  # Agents move
        ["move_down", "move_left", "pass"],  # Agents move and pass
        ["move_down", "move_left", "move_left"],  # Agents move
        ["pass", "move_left", "move_up"],  # Agents pass and move
        ["move_left", "move_up", "move_up"],  # Agents move
        ["move_up", "move_left", "move_up"],  # Agents move
        ["move_up", "move_left", "move_left"],  # Agents move
    ]

    for step, actions in enumerate(time_steps):
        rewards = environment.step(actions)
        print(f"\nStep {step + 1} - Actions: {actions}")
        for idx, reward in enumerate(rewards):
            print(f"Agent {idx + 1} - Reward: {reward}")
        environment.render()

        if environment.done:
            print("\nTask solved!")
            break

    print("Final state:")
    environment.print_state()
