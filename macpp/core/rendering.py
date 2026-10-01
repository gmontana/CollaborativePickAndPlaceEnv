"""
2D rendering of collaborative pick and place environment using PyGame
"""

import os

import pygame

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Viewer:
    def __init__(self, env):

        self.env = env

        self.offscreen_surface = pygame.Surface(
            (self.env.grid_width * self.env.cell_size, self.env.grid_length * self.env.cell_size)
        )

        self.screen = None

        # Pre-calculate the icon size based on the cell size
        self.icon_size = max(1, int(self.env.cell_size * 0.8))
        self.goal_label = None
        if self.env.cell_size >= 48:
            pygame.font.init()
            font = pygame.font.Font(None, max(16, self.env.cell_size // 6))
            self.goal_label = font.render("GOAL", True, (160, 35, 35))

        base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, "icons")

        # Load and resize images once during initialization
        self.picker_icon = self._load_and_resize_image(
            os.path.join(icon_path, "picker_not_carrying_large_new.png")
        )
        self.picker_carrying_icon = self._load_and_resize_image(
            os.path.join(icon_path, "picker_carrying_large_new.png")
        )
        self.non_picker_with_box_icon = self._load_and_resize_image(
            os.path.join(icon_path, "dropper_not_carrying_with_box_large.png")
        )
        self.non_picker_icon = self._load_and_resize_image(
            os.path.join(icon_path, "dropper_not_carrying_large.png")
        )
        self.non_picker_carrying_icon = self._load_and_resize_image(
            os.path.join(icon_path, "dropper_carrying_large.png")
        )
        self.object_icon = self._load_and_resize_image(os.path.join(icon_path, "box_large.png"))

    def _load_and_resize_image(self, image_path):
        try:
            image = pygame.image.load(image_path)
            return pygame.transform.scale(image, (self.icon_size, self.icon_size))
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Image file {image_path} not found.")

    def _grid_to_screen(self, grid_x, grid_y):
        """Convert grid coordinates to screen coordinates."""
        x = grid_x * self.env.cell_size + self.env.cell_size // 2
        y = grid_y * self.env.cell_size + self.env.cell_size // 2
        return x, y

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

    def _get_agent_icon(self, agent, can_display_side_by_side):
        if can_display_side_by_side:
            return self.non_picker_with_box_icon
        if agent.picker:
            return (
                self.picker_carrying_icon if agent.carrying_object is not None else self.picker_icon
            )
        return (
            self.non_picker_carrying_icon
            if agent.carrying_object is not None
            else self.non_picker_icon
        )

    def _draw_agent_icon(self, agent_icon, cell_center):
        agent_icon_rect = agent_icon.get_rect(center=cell_center)
        self.offscreen_surface.blit(agent_icon, agent_icon_rect)

    def _draw_rect(self, color, x, y, width, height, thickness=0):
        """Draw a rectangle on the offscreen surface."""
        rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
        pygame.draw.rect(self.offscreen_surface, color, rect, thickness)

    def _draw_agents(self):
        for agent in self.env.agents:
            x, y = self._grid_to_screen(*agent.position)
            object_at_position = any(obj.position == agent.position for obj in self.env.objects)
            can_display_side_by_side = (
                not agent.picker and agent.carrying_object is None and object_at_position
            )
            agent_icon = self._get_agent_icon(agent, can_display_side_by_side)
            self._draw_agent_icon(agent_icon, (x, y))

    def _draw_objects(self):
        for obj in self.env.objects:
            if obj.carrying_agent is not None:
                continue
            x, y = self._grid_to_screen(*obj.position)
            self.offscreen_surface.blit(
                self.object_icon, (x - self.icon_size // 2, y - self.icon_size // 2)
            )

    def _draw_goals(self):
        for goal in self.env.goals:
            x, y = self._grid_to_screen(*goal)
            delivered = any(
                obj.position == goal and obj.carrying_agent is None for obj in self.env.objects
            )
            border_color = GREEN if delivered else RED
            self._draw_rect(border_color, x, y, self.env.cell_size, self.env.cell_size, thickness=8)
            occupied = any(agent.position == goal for agent in self.env.agents) or any(
                obj.position == goal for obj in self.env.objects
            )
            if not occupied and self.goal_label is not None:
                self.offscreen_surface.blit(
                    self.goal_label, self.goal_label.get_rect(center=(x, y))
                )

    def render(self, mode="human"):
        self.offscreen_surface.fill(WHITE)
        self._draw_grid()
        self._draw_goals()
        self._draw_objects()
        self._draw_agents()
        if mode == "rgb_array":
            return pygame.surfarray.array3d(self.offscreen_surface).transpose(1, 0, 2).copy()
        if self.screen is None or not pygame.display.get_init():
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.offscreen_surface.get_size())
            pygame.display.set_caption("Collaborative Multi-Agent Pick and Place")
        self.screen.blit(self.offscreen_surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
