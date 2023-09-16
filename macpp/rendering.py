"""
    2D rendering of collaborative pick and place environment using PyGame
"""
import pygame
import os

# _ANIMATION_DELAY = 500
_ANIMATION_FPS = 5

_WHITE = (255, 255, 255)
_GRAY = (200, 200, 200)
_BLUE = (0, 0, 255)
_RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLACK = (0, 0, 0)
# LIGHT_GRAY = (200, 200, 200)
# YELLOW = (255, 255, 0)


class Viewer:
    def __init__(self, width, length, cell_size):
        self.width = width
        self.length = length
        self.cell_size = cell_size

        # Initialize pygame
        pygame.init()

        # Create the offscreen surface for rendering
        self.offscreen_surface = pygame.Surface(
            (self.width * self.cell_size, self.length * self.cell_size)
        )

        # Load agent icons
        base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, "icons")
        self.picker_icon = self._load_image(
            os.path.join(icon_path, "agent_picker.png"), _RED
        )
        self.non_picker_icon = self._load_image(
            os.path.join(icon_path, "agent_non_picker.png"), _BLUE
        )

        # Create the screen for display
        self.screen = pygame.display.set_mode(
            (self.width * self.cell_size, self.length * self.cell_size)
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
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.offscreen_surface,
                (0, 0, 0),
                (x, 0),
                (x, self.length * self.cell_size),
            )
        for y in range(0, self.length * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.offscreen_surface,
                (0, 0, 0),
                (0, y),
                (self.width * self.cell_size, y),
            )

    def _draw_agents(self, agents):
        for agent in agents:
            x, y = agent.position
            cell_center = (
                x * self.cell_size + self.cell_size // 2,
                y * self.cell_size + self.cell_size // 2,
            )
            scaling_factor = 0.8
            icon_size = int(self.cell_size * scaling_factor)

            try:
                agent_icon = self.picker_icon if agent.picker else self.non_picker_icon
                agent_icon_resized = pygame.transform.scale(
                    agent_icon, (icon_size, icon_size)
                )
                agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
                self.offscreen_surface.blit(agent_icon_resized, agent_icon_rect)

                if agent.carrying_object is not None:
                    thickness = 3
                    pygame.draw.rect(
                        self.offscreen_surface, (0, 255, 0), agent_icon_rect, thickness
                    )

            except Exception:
                # Fallback rendering
                color = (255, 0, 0) if agent.picker else (0, 0, 255)
                pygame.draw.rect(
                    self.offscreen_surface,
                    color,
                    (
                        x * self.cell_size + self.cell_size // 4,
                        y * self.cell_size + self.cell_size // 4,
                        self.cell_size // 2,
                        self.cell_size // 2,
                    ),
                )

    def _draw_objects(self, objects):
        for obj in objects:
            x, y = obj.position
            pygame.draw.circle(
                self.offscreen_surface,
                (0, 255, 0),
                (
                    x * self.cell_size + self.cell_size // 2,
                    y * self.cell_size + self.cell_size // 2,
                ),
                self.cell_size // 4,
            )

    def _draw_goals(self, goals):
        for goal in goals:
            x, y = goal
            pygame.draw.rect(
                self.offscreen_surface,
                _GRAY,
                (
                    x * self.cell_size + self.cell_size // 3,
                    y * self.cell_size + self.cell_size // 3,
                    self.cell_size // 3,
                    self.cell_size // 3,
                ),
            )

    def render(self, agents, objects, goals):
        self.offscreen_surface.fill(_WHITE)
        self._draw_grid()
        self._draw_agents(agents)
        self._draw_objects(objects)
        self._draw_goals(goals)
        self.screen.blit(self.offscreen_surface, (0, 0))
        pygame.display.flip()
        # pygame.time.wait(_ANIMATION_DELAY)

    def close(self):
        pygame.quit()

    # def render(self, agents, objects, goals):
    #     # Fill background
    #     self.offscreen_surface.fill((255, 255, 255))

    #     # Draw elements
    #     self._draw_grid()
    #     for agent in agents:
    #         self._draw_agent(agent)
    #     self._draw_objects(objects)

    #     # Update the display
    #     self.screen.blit(self.offscreen_surface, (0, 0))
    #     pygame.display.flip()

    # def close(self):
    #     pygame.quit()

    # def render(self):

    #     # Fill background
    #     self.offscreen_surface.fill(WHITE)

    #     # Draw grid
    #     for x in range(0, self.width * self.cell_size, self.cell_size):
    #         pygame.draw.line(
    #             self.offscreen_surface, BLACK, (x, 0), (x, self.length * self.cell_size)
    #         )
    #     for y in range(0, self.length * self.cell_size, self.cell_size):
    #         pygame.draw.line(
    #             self.offscreen_surface, BLACK, (0, y), (self.width * self.cell_size, y)
    #         )

    #     # Draw objects
    #     for obj in self.objects:
    #         x, y = obj.position
    #         pygame.draw.circle(
    #             self.offscreen_surface,
    #             GREEN,
    #             (
    #                 x * self.cell_size + self.cell_size // 2,
    #                 y * self.cell_size + self.cell_size // 2,
    #             ),
    #             self.cell_size // 4,
    #         )

    #     # Draw goals (small rectangles)
    #     for goal in self.goals:
    #         x, y = goal
    #         pygame.draw.rect(
    #             self.offscreen_surface,
    #             LIGHT_GRAY,
    #             (
    #                 x * self.cell_size + self.cell_size // 3,
    #                 y * self.cell_size + self.cell_size // 3,
    #                 self.cell_size // 3,
    #                 self.cell_size // 3,
    #             ),
    #         )

    #     # Load icons for agents
    #     base_path = os.path.dirname(__file__)
    #     icon_path = os.path.join(base_path, "icons")
    #     self.picker_icon = self._load_image(
    #         os.path.join(icon_path, "agent_picker.png"), (255, 0, 0)
    #     )
    #     self.non_picker_icon = self._load_image(
    #         os.path.join(icon_path, "agent_non_picker.png"), (0, 0, 255)
    #     )

    #     # Draw agents
    #     for agent in self.agents:
    #         x, y = agent.position
    #         cell_center = (
    #             x * self.cell_size + self.cell_size // 2,
    #             y * self.cell_size + self.cell_size // 2,
    #         )
    #         scaling_factor = 0.8
    #         icon_size = int(self.cell_size * scaling_factor)

    #         try:
    #             # Use icons
    #             agent_icon = self.picker_icon if agent.picker else self.non_picker_icon
    #             agent_icon_resized = pygame.transform.scale(
    #                 agent_icon, (icon_size, icon_size)
    #             )
    #             agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
    #             self.offscreen_surface.blit(agent_icon_resized, agent_icon_rect)

    #             # Agent is carrying an object
    #             if agent.carrying_object is not None:
    #                 thickness = 3
    #                 pygame.draw.rect(
    #                     self.offscreen_surface, GREEN, agent_icon_rect, thickness
    #                 )

    #         except Exception:
    #             # Fallback to default rendering using shapes and colors
    #             color = RED if agent.picker else BLUE
    #             if agent.carrying_object is not None:
    #                 pygame.draw.circle(
    #                     self.offscreen_surface, color, cell_center, self.cell_size // 3
    #                 )
    #                 pygame.draw.rect(
    #                     self.offscreen_surface,
    #                     YELLOW,
    #                     (
    #                         x * self.cell_size + self.cell_size // 3,
    #                         y * self.cell_size + self.cell_size // 3,
    #                         self.cell_size // 3,
    #                         self.cell_size // 3,
    #                     ),
    #                 )
    #             else:
    #                 pygame.draw.rect(
    #                     self.offscreen_surface,
    #                     color,
    #                     (
    #                         x * self.cell_size + self.cell_size // 4,
    #                         y * self.cell_size + self.cell_size // 4,
    #                         self.cell_size // 2,
    #                         self.cell_size // 2,
    #                     ),
    #                 )

    #     # If rendering is enabled, blit the offscreen surface to the screen and update the display
    #     if self.enable_rendering:
    #         self.screen.blit(self.offscreen_surface, (0, 0))
    #         pygame.display.flip()
    #         pygame.time.wait(ANIMATION_DE
