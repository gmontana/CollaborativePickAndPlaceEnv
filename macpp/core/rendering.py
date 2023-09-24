"""
2D rendering of collaborative pick and place environment using PyGame
"""
import pygame
import os

_ANIMATION_FPS = 5

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Viewer:
    def __init__(self, env):

        self.env = env

        # Initialize pygame
        pygame.init()

        # Create the offscreen surface for rendering
        self.offscreen_surface = pygame.Surface(
            (self.env.grid_width * self.env.cell_size, self.env.grid_length * self.env.cell_size)
        )

        # Load agent icons
        base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, "icons")
        self.picker_icon = self._load_image(
            os.path.join(icon_path, "agent_picker.png"), RED
        )
        self.non_picker_icon = self._load_image(
            os.path.join(icon_path, "agent_non_picker.png"), BLUE
        )

        # Create the screen for display
        self.screen = pygame.display.set_mode(
            (self.env.grid_width * self.env.cell_size, self.env.grid_length * self.env.cell_size)
        )
        pygame.display.set_caption("Collaborative Multi-Agent Pick and Place")

    def _load_image(self, image_path, placeholder_color):
        try:
            return pygame.image.load(image_path)
        except FileNotFoundError:
            print(
                f"Warning: Image file {image_path} not found. Using a default placeholder."
            )
            placeholder = pygame.Surface((50, 50))
            pygame.draw.rect(placeholder, placeholder_color, (0, 0, 50, 50))
            return placeholder

    def _draw_grid(self):
        for x in range(0, self.env.grid_width * self.env.cell_size, self.env.cell_size):
            pygame.draw.line(
                self.offscreen_surface,
                (0, 0, 0),
                (x, 0),
                (x, self.env.grid_length * self.env.cell_size),
            )
        for y in range(0, self.env.grid_length * self.env.cell_size, self.env.cell_size):
            pygame.draw.line(
                self.offscreen_surface,
                (0, 0, 0),
                (0, y),
                (self.env.grid_width * self.env.cell_size, y),
            )

    def _draw_agents(self):
        for agent in self.env.agents:
            x, y = agent.position
            cell_center = (
                x * self.env.cell_size + self.env.cell_size // 2,
                y * self.env.cell_size + self.env.cell_size // 2,
            )
            icon_size = int(self.env.cell_size * 0.8)

            try:
                agent_icon = self.picker_icon if agent.picker else self.non_picker_icon
                agent_icon_resized = pygame.transform.scale(
                    agent_icon, (icon_size, icon_size)
                )
                agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
                self.offscreen_surface.blit(agent_icon_resized, agent_icon_rect)

                # Draw rectangle around the agent carrying an object
                if agent.carrying_object is not None:
                    pygame.draw.rect(
                        self.offscreen_surface, GREEN, agent_icon_rect, 5
                    )

            except Exception:
                # Fallback rendering
                color = RED if agent.picker else BLUE
                pygame.draw.rect(
                    self.offscreen_surface,
                    color,
                    (
                        x * self.env.cell_size + self.env.cell_size // 4,
                        y * self.env.cell_size + self.env.cell_size // 4,
                        self.env.cell_size // 2,
                        self.env.cell_size // 2,
                    ),
                )

    def _draw_objects(self):
        for obj in self.env.objects:
            x, y = obj.position
            pygame.draw.circle(
                self.offscreen_surface,
                (0, 255, 0),
                (
                    x * self.env.cell_size + self.env.cell_size // 2,
                    y * self.env.cell_size + self.env.cell_size // 2,
                ),
                self.env.cell_size // 4,
            )

    def _draw_goals(self):
        for goal in self.env.goals:
            x, y = goal
            pygame.draw.rect(
                self.offscreen_surface,
                GRAY,
                (
                    x * self.env.cell_size + self.env.cell_size // 3,
                    y * self.env.cell_size + self.env.cell_size // 3,
                    self.env.cell_size // 3,
                    self.env.cell_size // 3,
                ),
            )

    def render(self):
        self.offscreen_surface.fill(WHITE)
        self._draw_grid()
        self._draw_objects()
        self._draw_goals()
        self._draw_agents()
        self.screen.blit(self.offscreen_surface, (0, 0))
        pygame.display.flip()

    def close(self):
        pygame.quit()
