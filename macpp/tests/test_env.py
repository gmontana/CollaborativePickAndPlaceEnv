import unittest
from ..core.environment import MACPPEnv, Action, REWARD_PICKUP, REWARD_COMPLETION, REWARD_STEP, REWARD_BAD_PASS, REWARD_GOOD_PASS, REWARD_DROP

DEBUG=False

def get_obs(obs):
    ''' Utility function to covert the format of the observations '''
    new_observation = {}
    agent_observations = {}
    
    for idx, agent in enumerate(obs["agents"]):
        agent_observations[f"agent_{idx}"] = {
            "position": agent["position"],
            "picker": agent["picker"],
            "carrying_object": agent["carrying_object"]
        }
    
    new_observation["agents"] = agent_observations
    new_observation["objects"] = tuple(obs["objects"])
    new_observation["goals"] = tuple(obs["goals"])
    
    return new_observation

class MACPPTests(unittest.TestCase):

    def test_pass_between_two_agents(self):
        initial_state = {
            "agents": [
                {
                    "position": (0, 0),
                    "picker": True,
                    "carrying_object": 0,
                },  # Picker Agent carrying Object with ID 0
                {
                    "position": (0, 1),
                    "picker": False,
                    "carrying_object": None,
                },  # Non-picker Agent not carrying any object
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(2,2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,4]

        expected_reward = 2* REWARD_GOOD_PASS + 2* REWARD_STEP
        _, reward, _, _ = env.step(actions)
        self.assertEqual(expected_reward, reward, "Incorrect reward for good pass.")

        # Check that the picker agent passed its object to the non-picker agent
        self.assertIsNone(
            env.agents[0].carrying_object, "Picker Agent did not pass the object."
        )
        self.assertEqual(
            env.agents[1].carrying_object,
            0,
            "Non-picker Agent did not receive the object.",
        )


    def test_invalid_moves(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (1, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(0, 0)],
        }

        env = MACPPEnv((10,10), n_agents=2, n_pickers=1, initial_state=get_obs(initial_state), debug_mode=DEBUG)

        actions = [3,2]
        env.step(actions)
        self.assertEqual(
            env.agents[0].position,
            (0, 0),
            "Agent moved out of bounds or overlapped with another agent.",
        )

    def test_dropoff_object(self):
        initial_state = {
            "agents": [
                {"position": (2, 2), "picker": False, "carrying_object": 0},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(1, 2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [2,5]
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
            "agents": [
                {"position": (1, 1), "picker": True, "carrying_object": None},
                {"position": (8, 8), "picker": False, "carrying_object": None},
                ],
            "objects": [{"position": (1, 2), "id": 0}],
            "goals": [(1,2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [1,5]
        env.step(actions)
        self.assertEqual(
            env.agents[0].carrying_object,
            0,
            "Picker agent failed to pick up the object.",
        )

    def test_non_picker_agent_trying_to_pick_up_object(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": False, "carrying_object": None},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,5]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object,
            "Non-picker agent managed to pick up the object.",
        )

    def test_object_disappearing_after_being_picked_up(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,5]
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
        env = MACPPEnv((10, 10), 3, 2, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,4,4]
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
            "agents": [
                {"position": (0, 0), "picker": False, "carrying_object": None},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [],
            "goals": [(1, 0)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [3,5]
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
        env = MACPPEnv((10, 10), n_agents=2, n_pickers=1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        # First pass the object from picker to non picker
        actions = [4,4]
        env.step(actions)
        # Then move to goal position
        actions = [1,1]
        _, _, done, _ = env.step(actions)
        self.assertTrue(env.done, "Termination condition not recognized.")

    def test_pass_object_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(0,2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,4]
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
                {"position": (0, 1), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,2]
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
                {"position": (1, 0), "picker": False, "carrying_object": 1},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (1, 0), "id": 1}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,4]
        env.step(actions)
        self.assertTrue(
            env.agents[0].carrying_object is not None
            and env.agents[1].carrying_object is not None,
            "Agents failed to handle multiple passes.",
        )

    def test_invalid_pass(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,5]
        env.step(actions)
        self.assertEqual(
            env.agents[0].carrying_object,
            0,
            "Agent passed the object when it was not valid.",
        )

    def test_non_picker_cannot_pick(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": False, "carrying_object": None},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,5]
        env.step(actions)
        self.assertIsNone(
            env.agents[0].carrying_object,
            "Agent that's not a picker managed to pick an object.",
        )

    def test_simultaneous_pass_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": 1},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (1, 0), "id": 1}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [4,4]
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
        env = MACPPEnv((10, 10), 4, 2, initial_state=get_obs(initial_state), debug_mode=DEBUG)

        actions = [4,4,4,4]
        env.step(actions)

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
            "agents": [
                {"position": (1, 1), "picker": False, "carrying_object": 0},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [3,5]
        env.step(actions)
        # Check that the object is still in the environment after being dropped on a goal
        self.assertTrue(
            any(obj for obj in env.objects if obj.id == 0),
            "Object disappeared after being dropped on a goal by a non-picker agent.",
        )

    def test_picker_drop_on_goal(self):
        initial_state = {
            "agents": [
                {"position": (1, 1), "picker": True, "carrying_object": 0},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
                ],
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(initial_state), debug_mode=DEBUG)
        actions = [3,5]
        env.step(actions)
        # Check that the object is still in the environment after being dropped on a goal by a picker agent
        self.assertTrue(
            any(obj for obj in env.objects if obj.id == 0),
            "Object disappeared after being dropped on a goal by a picker agent.",
        )


if __name__ == "__main__":

    unittest.main()
