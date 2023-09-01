import pygame
import random
import numpy as np
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK = 4
    PLACE = 5

class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agent_position = [0, 0]
        self.object_position = [random.randint(0, width - 1), random.randint(0, height - 1)]
        self.target_position = [random.randint(0, width - 1), random.randint(0, height - 1)]
        self.object_held = False

    def reset(self):
        self.agent_position = [0, 0]
        self.object_position = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.target_position = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
        self.object_held = False
        return self.get_state()

    def step(self, action):
        if action == Action.UP:
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)
        elif action == Action.DOWN:
            self.agent_position[1] = min(self.agent_position[1] + 1, self.height - 1)
        elif action == Action.LEFT:
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        elif action == Action.RIGHT:
            self.agent_position[0] = min(self.agent_position[0] + 1, self.width - 1)
        elif action == Action.PICK:
            if self.agent_position == self.object_position and not self.object_held:
                self.object_held = True
                self.object_position = [-1, -1]
        elif action == Action.PLACE:
            if self.agent_position == self.target_position and self.object_held:
                self.object_held = False
                self.object_position = self.target_position
                self.target_position = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]

        reward = 0
        done = False
        if self.agent_position == self.object_position:
            reward = 1
        elif self.agent_position == self.target_position and self.object_held:
            reward = 10
            done = True

        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array([
            self.agent_position[0],
            self.agent_position[1],
            self.object_position[0],
            self.object_position[1],
            self.target_position[0],
            self.target_position[1],
            int(self.object_held),
        ])


def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))

    env = GridWorld(10, 10)
    cell_size = 50

    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state, reward, done, _ = env.step(random.choice(list(Action)))

        for i in range(env.width):
            for j in range(env.height):
                pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size), 1)

        agent_x, agent_y = env.agent_position
        pygame.draw.circle(screen, (255, 0, 0), (agent_x * cell_size + cell_size // 2, agent_y * cell_size + cell_size // 2), cell_size // 4)

        object_x, object_y = env.object_position
        if not env.object_held:
            pygame.draw.circle(screen, (0, 255, 0), (object_x * cell_size + cell_size // 2, object_y * cell_size + cell_size // 2), cell_size // 4)

        target_x, target_y = env.target_position
        pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(target_x * cell_size, target_y * cell_size, cell_size, cell_size), 5)

        pygame.display.flip()

        if done:
            env.reset()

    pygame.quit()

if __name__ == "__main__":
    main()

