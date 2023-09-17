import unittest
from ..core.environment import MultiAgentPickAndPlace
from ..core.environment import Action

DEBUG=True

class MACPPTests(unittest.TestCase):

    def test_invalid_moves(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (1, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(0, 0)],
        }

        env = MultiAgentPickAndPlace(width=10, length=10, n_agents=2, n_pickers=0, initial_state=initial_state, debug_mode=DEBUG)

        actions = [Action.RIGHT, Action.LEFT]
        env.step(actions)
        self.assertEqual(
            env.agents[0].position,
            (0, 0),
            "Agent moved out of bounds or overlapped with another agent.",
        )

    def test_dropoff_object(self):
        initial_state = {
            "agents": [{"position": (2, 2), "picker": False, "carrying_object": 0}],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(1, 2)],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.LEFT]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object, "Agent failed to drop off the object."
        )
        self.assertEqual(
            env.objects[0].position,
            (1, 2),
            "Object was not dropped at the correct position.",
        )

    def test_picker_agent_picking_up_object(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": True, "carrying_object": None}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS]
        env.step(actions)
        self.assertEqual(
            env.agents[0].carrying_object,
            0,
            "Picker agent failed to pick up the object.",
        )

    def test_non_picker_agent_trying_to_pick_up_object(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": False, "carrying_object": None}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object,
            "Non-picker agent managed to pick up the object.",
        )

    def test_object_disappearing_after_being_picked_up(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": True, "carrying_object": None}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS]
        env.step(actions)
        self.assertNotIn(
            {"position": (0, 0), "id": 0},
            env.objects,
            "Object did not disappear after being picked up.",
        )

    def test_multiple_agents_trying_to_pass_to_same_agent(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (0, 1), "picker": True, "carrying_object": 1},
                {"position": (1, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (0, 1), "id": 1}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 3, 2, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.PASS, Action.PASS]
        env.step(actions)
        self.assertTrue(
            env.agents[2].carrying_object in [0, 1], "Agent did not receive an object."
        )
        self.assertTrue(
            sum([1 for agent in env.agents if agent.carrying_object is not None]) == 2,
            "More than one object was passed to the same agent.",
        )

    def test_agent_moving_to_goal_without_object(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": False, "carrying_object": None}],
            "objects": [],
            "goals": [(1, 0)],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 0, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.RIGHT]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object,
            "Agent status changed even without carrying an object.",
        )

    def test_termination_condition(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (0, 1), "picker": False, "carrying_object": None},
            ],
            "objects": [
                {"position": (0, 0), "id": 0},
            ],
            "goals": [(0, 2)]
        }
        env = MultiAgentPickAndPlace(10, 10, n_agents=2, n_pickers=1, initial_state=initial_state, debug_mode=DEBUG)
        # First pass the object from picker to non picker
        actions = [Action.PASS, Action.PASS]
        env.step(actions)
        # Then move to goal position
        actions = [Action.DOWN, Action.DOWN]
        _, _, done, _ = env.step(actions)
        self.assertTrue(env.done, "Termination condition not recognized.")

    def test_pass_object_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.PASS]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object, "Agent 1 did not pass the object"
        )
        self.assertEqual(
            env.agents[1].carrying_object, 0, "Agent 2 did not receive the object"
        )

    def test_multiple_agents_pick_same_object(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (0, 1), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 2, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.LEFT]
        env.step(actions)
        self.assertTrue(
            env.agents[0].carrying_object is not None
            or env.agents[1].carrying_object is not None,
            "Multiple agents picked up the same object.",
        )
        self.assertNotEqual(
            env.agents[0].carrying_object,
            env.agents[1].carrying_object,
            "Both agents managed to pick up the object.",
        )

    def test_multiple_agent_passes(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": 1},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (1, 0), "id": 1}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 2, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.PASS]
        env.step(actions)
        self.assertTrue(
            env.agents[0].carrying_object is not None
            and env.agents[1].carrying_object is not None,
            "Agents failed to handle multiple passes.",
        )

    def test_invalid_pass(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": True, "carrying_object": 0}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS]
        env.step(actions)
        self.assertEqual(
            env.agents[0].carrying_object,
            0,
            "Agent passed the object when it shouldn't have.",
        )

    def test_non_picker_cannot_pick(self):
        initial_state = {
            "agents": [{"position": (0, 0), "picker": False, "carrying_object": None}],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object,
            "Agent that's not a picker managed to pick an object.",
        )

    def test_simultaneous_pass_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": 1},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (1, 0), "id": 1}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 2, 2, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.PASS]
        env.step(actions)
        # Check that both agents still have their objects after trying to pass to each other
        self.assertEqual(env.agents[0].carrying_object, 0, "Agent 1 lost its object.")
        self.assertEqual(env.agents[1].carrying_object, 1, "Agent 2 lost its object.")

    def test_complex_pass_between_four_agents(self):
        initial_state = {
            "agents": [
                {
                    "position": (0, 0),
                    "picker": True,
                    "carrying_object": 0,
                },  # Picker Agent 1
                {
                    "position": (0, 1),
                    "picker": True,
                    "carrying_object": 1,
                },  # Picker Agent 2
                {
                    "position": (1, 0),
                    "picker": False,
                    "carrying_object": None,
                },  # Non-picker Agent 1
                {
                    "position": (1, 1),
                    "picker": False,
                    "carrying_object": None,
                },  # Non-picker Agent 2
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (0, 1), "id": 1}],
            "goals": [],
        }
        env = MultiAgentPickAndPlace(10, 10, 4, 2, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.PASS, Action.PASS, Action.PASS, Action.PASS]
        env.step(actions)
        # Check that the picker agents passed their objects to the non-picker agents
        self.assertIsNone(
            env.agents[0].carrying_object, "Picker Agent 1 did not pass the object."
        )
        self.assertIsNone(
            env.agents[1].carrying_object, "Picker Agent 2 did not pass the object."
        )
        self.assertEqual(
            env.agents[2].carrying_object,
            0,
            "Non-picker Agent 1 did not receive the object.",
        )
        self.assertEqual(
            env.agents[3].carrying_object,
            1,
            "Non-picker Agent 2 did not receive the object.",
        )

    def test_non_picker_drop_on_goal(self):
        initial_state = {
            "agents": [{"position": (1, 1), "picker": False, "carrying_object": 0}],
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.RIGHT]
        env.step(actions)
        # Check that the object is still in the environment after being dropped on a goal
        self.assertTrue(
            any(obj for obj in env.objects if obj.id == 0),
            "Object disappeared after being dropped on a goal by a non-picker agent.",
        )

    def test_picker_drop_on_goal(self):
        initial_state = {
            "agents": [{"position": (1, 1), "picker": True, "carrying_object": 0}],
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }
        env = MultiAgentPickAndPlace(10, 10, 1, 1, initial_state=initial_state, debug_mode=DEBUG)
        actions = [Action.RIGHT]
        env.step(actions)
        # Check that the object is still in the environment after being dropped on a goal by a picker agent
        self.assertTrue(
            any(obj for obj in env.objects if obj.id == 0),
            "Object disappeared after being dropped on a goal by a picker agent.",
        )


if __name__ == "__main__":

    unittest.main()
