import unittest
from gym import spaces
from surviveworld import SurviveWorld

class SurviveWorldTestCase(unittest.TestCase):
    def setUp(self):
        self.env = SurviveWorld(grid_size=3, num_agents=2, initial_reward=10.0)

    def test_reset(self):
        observation = self.env.reset()
        self.assertEqual(len(observation), 8)  # Check if the observation has the correct length

    def test_step(self):
        self.env.reset()
        actions = [0, 1]  # Example actions for the agents
        observation, rewards, done, _ = self.env.step(actions)
        self.assertEqual(len(observation), 3)  # Check if the observation has the correct length
        self.assertEqual(len(rewards), 2)  # Check if the rewards have the correct length
        self.assertIsInstance(done, bool)  # Check if done is a boolean

    def test_action_space(self):
        self.assertIsInstance(self.env.action_space, spaces.MultiDiscrete)  # Check if the action space is of the correct type
        self.assertEqual(self.env.action_space.shape, (2,))  # Check the shape of the action space

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space, spaces.Tuple)  # Check if the observation space is of the correct type
        self.assertEqual(len(self.env.observation_space.spaces), 8)  # Check the number of spaces in the observation space
    
if __name__ == '__main__':
    unittest.main()