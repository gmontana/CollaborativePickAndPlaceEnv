import unittest
from ..core.environment import MACPPEnv, Action, REWARD_PICKUP, REWARD_COMPLETION, REWARD_STEP, REWARD_BAD_PASS, REWARD_GOOD_PASS, REWARD_DROP

import gym

DEBUG = False


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

    def test_agents_swapping_places(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (0, 1), "picker": False, "carrying_object": None},
            ],
            "objects": [],
            "goals": [],
        }
        env = MACPPEnv((10, 10), n_agents=2, n_pickers=1,
                       initial_state=get_obs(initial_state), debug_mode=True)
        actions = [3, 2]
        env.step(actions)

        # After the step, agents should have swapped places
        self.assertEqual(env.agents[0].position, (0, 1),
                         "Agent 0 did not move to the right position.")
        self.assertEqual(env.agents[1].position, (0, 0),
                         "Agent 1 did not move to the left position.")

    '''
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
            "goals": [(2, 2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 4]

        expected_reward = 2 * REWARD_GOOD_PASS + 2 * REWARD_STEP
        _, reward, _, _ = env.step(actions)
        self.assertEqual(expected_reward, reward,
                         "Incorrect reward for good pass.")

        # Check that the picker agent passed its object to the non-picker agent
        self.assertIsNone(env.agents[0].carrying_object,
                          "Picker Agent did not pass the object.")
        self.assertEqual(env.agents[1].carrying_object.id if env.agents[1].carrying_object else None,
                         0, "Non-picker Agent did not receive the object.")

    def test_invalid_moves(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (1, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (2, 2), "id": 0}],
            "goals": [(0, 0)],
        }

        env = MACPPEnv((10, 10), n_agents=2, n_pickers=1,
                       initial_state=get_obs(initial_state), debug_mode=DEBUG)

        actions = [3, 2]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [2, 5]
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
            "goals": [(1, 2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [1, 3]
        env.step(actions)

        self.assertIsNotNone(
            env.agents[0].carrying_object,
            "Picker agent failed to pick up the object.",
        )
        self.assertEqual(
            env.agents[0].carrying_object.id,
            0,
            "Picker agent is carrying the wrong object.",
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 5]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 5]
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
        env = MACPPEnv((10, 10), 3, 2, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 4, 4]
        env.step(actions)

        self.assertTrue(
            env.agents[2].carrying_object.id in [
                0, 1] if env.agents[2].carrying_object else False, "Agent did not receive an object."
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [3, 5]
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
        env = MACPPEnv((10, 10), n_agents=2, n_pickers=1,
                       initial_state=get_obs(initial_state), debug_mode=DEBUG)
        # First pass the object from picker to non picker
        actions = [4, 4]
        env.step(actions)
        # Then move to goal position
        actions = [1, 1]
        _, _, done, _ = env.step(actions)
        self.assertTrue(env.done, "Termination condition not recognized.")

    def test_pass_object_between_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(0, 2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 4]
        env.step(actions)
        self.assertIsNone(env.agents[0].carrying_object,
                          "Agent 1 did not pass the object")
        self.assertEqual(env.agents[1].carrying_object.id if env.agents[1]
                         .carrying_object else None, 0, "Agent 2 did not receive the object")

    def test_multiple_agents_pick_same_object(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": None},
                {"position": (0, 1), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 2]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 4]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 5]
        env.step(actions)

        self.assertEqual(
            env.agents[0].carrying_object.id if env.agents[0].carrying_object else None,
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 5]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [4, 4]
        env.step(actions)

        # Check that both agents still have their objects after trying to pass to each other
        self.assertEqual(
            env.agents[0].carrying_object.id if env.agents[0].carrying_object else None, 0, "Agent 1 lost its object.")
        self.assertEqual(
            env.agents[1].carrying_object.id if env.agents[1].carrying_object else None, 1, "Agent 2 lost its object.")

    def test_non_picker_drop_on_goal(self):
        initial_state = {
            "agents": [
                {"position": (1, 1), "picker": False, "carrying_object": 0},
                {"position": (8, 8), "picker": True, "carrying_object": 0},
            ],
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [3, 5]
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
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)
        actions = [3, 5]
        env.step(actions)
        # Check that the object is still in the environment after being dropped on a goal by a picker agent
        self.assertTrue(
            any(obj for obj in env.objects if obj.id == 0),
            "Object disappeared after being dropped on a goal by a picker agent.",
        )

    def test_movement_rewards(self):
        # Define the initial state with two agents
        initial_state = {
            "agents": {
                "0": {"position": (0, 0), "picker": True, "carrying_object": None},
                "1": {"position": (1, 0), "picker": False, "carrying_object": None},
            },
            "objects": [{"position": (1, 1), "id": 0}],
            "goals": [(2, 1)],
        }

        env = MACPPEnv(grid_size=(5, 5), n_agents=2, n_pickers=1,
                       initial_state=initial_state, debug_mode=DEBUG)
        actions = [1, 3]
        _, reward, _, _ = env.step(actions)
        expected_reward = 2 * REWARD_STEP
        self.assertEqual(reward, expected_reward,
                         "Incorrect reward for moving.")

    def test_env_settings(self):
        env_settings = [
            'macpp-10x10-2a-1p-2o-v0',
            'macpp-10x10-4a-2p-3o-v0',
            'macpp-10x10-4a-2p-2o-v0',
            'macpp-10x10-4a-3p-3o-v0'
        ]

        for setting in env_settings:
            env = gym.make(setting, debug_mode=DEBUG)
            obs, _ = env.reset()
            obs_space = env.observation_space

            # Check observation spaces are correct
            self.assertEqual(len(obs), len(
                obs_space), f"Observation and observation space lengths do not match for {setting}")

            # Take 10 steps with random actions
            for _ in range(10):
                actions = list(env.action_space.sample())
                obs, reward, done, info = env.step(actions)
                if done:
                    break

    def test_another_pass_between_two_agents(self):
        # Initialize the environment for this specific test case
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
            "goals": [(2, 2)],
        }
        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)

        actions = [4, 4]

        expected_reward = 2 * REWARD_GOOD_PASS + 2 * REWARD_STEP
        _, reward, _, _ = env.step(actions)
        self.assertEqual(expected_reward, reward,
                         "Incorrect reward for good pass.")

        # Check that the picker agent passed its object to the non-picker agent
        self.assertIsNone(
            env.agents[0].carrying_object, "Picker Agent did not pass the object."
        )
        self.assertEqual(
            env.agents[1].carrying_object.id if env.agents[1].carrying_object else None,
            0,
            "Non-picker Agent did not receive the object.",
        )

    def test_pass_between_three_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": True, "carrying_object": None},
                {"position": (2, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(2, 2)],
        }
        env = MACPPEnv((10, 10), 3, 2, initial_state=get_obs(
            initial_state), debug_mode=DEBUG)

        actions = [4, 4, 4]
        env.step(actions)

        # Check that Agent 1 received the object
        self.assertEqual(
            env.agents[1].carrying_object.id if env.agents[1].carrying_object else None, 0, "Non-picker Agent 1 did not receive the object."
        )

    def test_pass_two_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (2, 0), "id": 1}],
            "goals": [(4, 2)],
        }

        print(f"Initial state: {initial_state}")

        env = MACPPEnv((10, 10), 2, 1, initial_state=get_obs(
            initial_state), debug_mode=True)

        actions = [4, 4]
        new_state, _, _, _ = env.step(actions)

        print(f"New state: {new_state}")

        self.assertEqual(
            env.agents[1].carrying_object.id if env.agents[1].carrying_object else None, 0, "Agent 1 did not receive the object from Agent 0."
        )

    def test_pass_with_four_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
                {"position": (2, 0), "picker": False, "carrying_object": None},
                {"position": (3, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(4, 0)],
        }

        env = MACPPEnv((10, 10), 4, 1, initial_state=get_obs(
            initial_state), debug_mode=False)
        actions = [4, 4, 4, 4]

        new_state, _, _, _ = env.step(actions)

        self.assertIsNone(env.agents[0].carrying_object,
                          "Picker Agent did not pass the object.")
        self.assertEqual(env.agents[1].carrying_object.id if env.agents[1].carrying_object else None,
                         0, "First Non-picker Agent did not receive the object.")
        self.assertIsNone(env.agents[2].carrying_object,
                          "Second Non-picker Agent should not have the object.")
        self.assertIsNone(env.agents[3].carrying_object,
                          "Third Non-picker Agent should not have the object.")

    def test_pass_with_three_agents(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
                {"position": (2, 0), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}],
            "goals": [(3, 0)],
        }

        env = MACPPEnv((10, 10), 3, 1, initial_state=get_obs(
            initial_state), debug_mode=False)
        actions = [4, 4, 4]
        new_state, _, _, _ = env.step(actions)

        self.assertIsNone(env.agents[0].carrying_object,
                          "Agent 0 should not be carrying the object.")
        self.assertEqual(env.agents[1].carrying_object.id if env.agents[1].carrying_object else None,
                         0, "Agent 1 should be carrying the object.")
        self.assertIsNone(env.agents[2].carrying_object,
                          "Agent 2 should not be carrying the object.")

        expected_reward_agent_0 = REWARD_GOOD_PASS + REWARD_STEP
        expected_reward_agent_1 = REWARD_GOOD_PASS + REWARD_STEP
        expected_reward_agent_2 = REWARD_STEP

        self.assertEqual(env.agents[0].reward, expected_reward_agent_0,
                         "Agent 0 should receive a bad pass reward.")
        self.assertEqual(env.agents[1].reward, expected_reward_agent_1,
                         "Agent 1 should be penalized for movement.")
        self.assertEqual(env.agents[2].reward, expected_reward_agent_2,
                         "Agent 2 should be penalized for movement.")

    def test_five_agents_single_pass(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
                {"position": (2, 0), "picker": True, "carrying_object": 1},
                {"position": (3, 0), "picker": False, "carrying_object": None},
                {"position": (4, 0), "picker": True, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (2, 0), "id": 1}],
            "goals": [(5, 5), (6, 6)],
        }

        print(f"Initial state: {initial_state}")

        env = MACPPEnv((10, 10), 5, 2, initial_state=get_obs(
            initial_state), debug_mode=False)
        actions = [4, 4]
        new_state, _, _, _ = env.step(actions)

        # Updated expected positions, carrying objects, and rewards for each agent involved in the test
        expected_outcomes = [
            {"position": (0, 0), "carrying_object": None,
             "picker": True, "reward": 4},
            {"position": (1, 0), "carrying_object": 0,
             "picker": False, "reward": 4}
        ]

        # Loop through only the agents involved in the test (assumed to be the first two agents)
        for i in range(2):
            agent = env.agents[i]
            carrying_object_id = agent.carrying_object.id if agent.carrying_object else None
            with self.subTest(agent=f"Agent {i}"):
                self.assertEqual(
                    agent.position, expected_outcomes[i]["position"])
                self.assertEqual(carrying_object_id,
                                 expected_outcomes[i]["carrying_object"])
                self.assertEqual(agent.picker, expected_outcomes[i]["picker"])
                self.assertEqual(agent.reward, expected_outcomes[i]["reward"])

    def test_three_agents_simultaneous_pass(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
                {"position": (2, 0), "picker": True, "carrying_object": 1},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (2, 0), "id": 1}],
            "goals": [(5, 5), (6, 6)],
        }

        print(f"Initial state: {initial_state}")

        env = MACPPEnv((10, 10), 3, 2, initial_state=get_obs(
            initial_state), debug_mode=False)
        actions = [4, 4, 4]
        new_state, _, _, _ = env.step(actions)

        expected_outcomes = [
            {"position": (0, 0), "carrying_object": None, "picker": True},
            {"position": (1, 0), "carrying_object": 0, "picker": False},
            {"position": (2, 0), "carrying_object": 1, "picker": True}
        ]

        for i in range(3):
            agent = env.agents[i]
            carrying_object_id = agent.carrying_object.id if agent.carrying_object else None
            with self.subTest(agent=f"Agent {i}"):
                self.assertEqual(
                    agent.position, expected_outcomes[i]["position"])
                self.assertEqual(carrying_object_id,
                                 expected_outcomes[i]["carrying_object"])
                self.assertEqual(agent.picker, expected_outcomes[i]["picker"])

    def test_five_agents_two_passes(self):
        initial_state = {
            "agents": [
                {"position": (0, 0), "picker": True, "carrying_object": 0},
                {"position": (1, 0), "picker": False, "carrying_object": None},
                {"position": (2, 0), "picker": True, "carrying_object": 1},
                {"position": (3, 0), "picker": False, "carrying_object": None},
            ],
            "objects": [{"position": (0, 0), "id": 0}, {"position": (2, 0), "id": 1}],
            "goals": [(5, 5), (6, 6)],
        }

        env = MACPPEnv((10, 10), 5, 2, initial_state=get_obs(
            initial_state), debug_mode=False)
        actions = [4, 4, 4, 4]  # All agents are trying to pass
        new_state, _, _, _ = env.step(actions)

        expected_outcomes = [
            {"position": (0, 0), "carrying_object": None,
             "picker": True, "reward": 4},
            {"position": (1, 0), "carrying_object": 0,
             "picker": False, "reward": 4},
            {"position": (2, 0), "carrying_object": None,
             "picker": True, "reward": 4},
            {"position": (3, 0), "carrying_object": 1,
             "picker": False, "reward": 4}
        ]

        for i in range(4):
            agent = env.agents[i]
            carrying_object_id = agent.carrying_object.id if agent.carrying_object else None
            with self.subTest(agent=f"Agent {i}"):
                self.assertEqual(
                    agent.position, expected_outcomes[i]["position"])
                self.assertEqual(carrying_object_id,
                                 expected_outcomes[i]["carrying_object"])
                self.assertEqual(agent.picker, expected_outcomes[i]["picker"])
                self.assertEqual(agent.reward, expected_outcomes[i]["reward"])

    '''


if __name__ == "__main__":

    unittest.main()
