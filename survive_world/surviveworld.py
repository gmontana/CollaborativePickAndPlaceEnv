from tkinter import W
import gymnasium as gym
from gym import spaces
import numpy as np
import random
import pygame

class SurviveWorld(gym.Env):
    def __init__(self, grid_size, num_agents, initial_reward):
        super(SurviveWorld, self).__init__()
        
        pygame.init()

        self.directions = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])  # right, left, up, down
        self.num_agents = num_agents
        self.initial_reward =  initial_reward
        self.rewards = np.full(num_agents, initial_reward)

        # rendering 
        self.grid_size = grid_size 
        self.block_size = 100
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size + 100  # Extra space for text
        self.screen = pygame.display.set_mode((self.width, self.height))

        # we use white for agent 1, blue for agent 2, red for food for agent 1, pink for food for agent 2
        self.colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (255, 105, 180)]  

        # visualise reward 
        pygame.font.init() 
        self.font = pygame.font.SysFont(None, 25)

        # Action space: 0:Up, 1:Down, 2:Left, 3:Right, 4:Pass, 5:Eat
        self.action_space = spaces.MultiDiscrete([6]*num_agents)
        
        # State: positions of agents, food color and position, rewards
        self.observation_space = spaces.Tuple((
            spaces.Discrete(grid_size), # Agent 1 X position
            spaces.Discrete(grid_size), # Agent 1 Y position
            spaces.Discrete(grid_size), # Agent 2 X position
            spaces.Discrete(grid_size), # Agent 2 Y position
            spaces.Discrete(2), # Food color
            spaces.Discrete(grid_size), # Food X position
            spaces.Discrete(grid_size), # Food Y position
            spaces.Box(low=0, high=1, shape=(num_agents,), dtype=np.float32), # Rewards
        ))
        
        # Initialize state
        self.reset()
    
    def reset(self):
        self.agent_pos = np.random.randint(self.grid_size, size=(self.num_agents, 2))
        self.food_pos = np.random.randint(self.grid_size, size=2)
        self.food_color = np.random.randint(2)
        self.rewards = np.full(self.num_agents, self.initial_reward)  
        return self._get_obs()

    def _is_valid_position(self, pos):
        if pos[0] >= 0 and pos[0] < self.grid_size and pos[1] >= 0 and pos[1] < self.grid_size:
            return True
        else:
            return False

    def _spawn_food(self):
        while True:
            new_food_pos = np.random.randint(self.grid_size, size=2)
            if (np.array_equal(new_food_pos, self.agent_pos[0]) or np.array_equal(new_food_pos, self.agent_pos[1])):
                continue  # If the new food position matches an agent's position, generate a new position
            if np.all(self.agent_pos[self.food_color] == self.food_pos):
                self.food_color = 1 - self.food_color  # Change food color to match the agent already on the food
            self.food_pos = new_food_pos
            break

    def _get_obs(self):
        return (self.agent_pos[0, 0], self.agent_pos[0, 1], self.agent_pos[1, 0], self.agent_pos[1, 1], 
                self.food_color, self.food_pos[0], self.food_pos[1], self.rewards)

    def _get_observation(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3))
        grid[self.agent_pos[0][0], self.agent_pos[0][1], 0] = 1
        grid[self.agent_pos[1][0], self.agent_pos[1][1], 1] = 1
        grid[self.food_pos[0], self.food_pos[1], 2] = 1
        return grid
    
    def step(self, actions):
        assert len(actions) == self.num_agents, "Number of actions should be equal to the number of agents"
        done = False

        for i, action in enumerate(actions):
            assert action in list(range(6)), f"Action {action} is not valid"
            new_pos = np.array(self.agent_pos[i])

            # Handle movement actions
            if action < 4:  # Movement action
                new_pos_temp = new_pos + self.directions[action]
                occupied_positions = [list(self.agent_pos[j]) for j in range(self.num_agents) if j != i]
                if self._is_valid_position(new_pos_temp) and not any(
                    list(new_pos_temp) in occupied_positions for pos in occupied_positions
                ):
                    self.agent_pos[i] = new_pos_temp
                    self.rewards[i] -= 0.1
                    new_pos = new_pos_temp

            # Handle food-related actions
            if np.all(self.agent_pos[i] == self.food_pos):
                if action == 4:  # Eat action
                    if self.food_color == i:
                        self.rewards[i] += 1  # Add +1 reward due to eating
                        self._spawn_food()  # Spawn new food after eating
                        print("Eaten!")
                elif action == 5:  # Pass action
                    if self.food_color == i and np.all(self.agent_pos[i] == self.food_pos):
                        self.food_color = 1 - i  # Change food color and keep the food in the grid
                        print("Passed!")

            # Check if the agent has run out of rewards, in which case the game is over
            if self.rewards[i] <= 0:
                done = True

        return self._get_observation(), self.rewards, done, {} 

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))

        cell_size = self.width // self.grid_size

        # Draw agents (squares)
        for i, pos in enumerate(self.agent_pos):
            pygame.draw.rect(self.screen, self.colors[i], 
                            pygame.Rect(pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size))

        # Draw food (circles)
        food_color = self.colors[self.food_color]  # Match food color with agent color
        food_radius = cell_size // 2
        food_center_x = self.food_pos[0] * cell_size + cell_size // 2
        food_center_y = self.food_pos[1] * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, food_color, (food_center_x, food_center_y), food_radius)

        # Draw rewards
        for i, reward in enumerate(self.rewards):
            reward_text = f'Agent {i+1} reward: {reward:.3f}'
            text = self.font.render(reward_text, True, self.colors[i])
            self.screen.blit(text, (10, self.grid_size * cell_size + i * 35))  

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        pygame.time.wait(200)  # Add delay for easier viewing
 

if __name__ == '__main__':
    env = SurviveWorld(grid_size=3, num_agents=2, initial_reward=4.0)

    for i_episode in range(10):
        observation = env.reset()
        for t in range(100):
            actions = [env.action_space.sample()[i] for i in range(env.num_agents)]  # Taking random actions
            observation, reward, done, info = env.step(actions)
            env.render()
            # print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    pygame.quit()