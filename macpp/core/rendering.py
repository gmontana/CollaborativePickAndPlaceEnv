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
            (self.env.grid_width * self.env.cell_size,
             self.env.grid_length * self.env.cell_size)
        )

        # Load all icons
        base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, "icons")
        self.picker_icon = self._load_image(
            os.path.join(icon_path, "picker_not_carrying.png"))
        self.picker_carrying_icon = self._load_image(
            os.path.join(icon_path, "picker_carrying.png"))
        self.non_picker_with_box_icon = self._load_image(
            os.path.join(icon_path, "dropper_not_carrying_with_box.png"))
        self.non_picker_icon = self._load_image(
            os.path.join(icon_path, "dropper_not_carrying.png"),)
        self.non_picker_carrying_icon = self._load_image(
            os.path.join(icon_path, "dropper_carrying.png"))
        self.object_icon = self._load_image(
            os.path.join(icon_path, "box.png"))

        # Create the screen for display
        self.screen = pygame.display.set_mode(
            (self.env.grid_width * self.env.cell_size,
             self.env.grid_length * self.env.cell_size)
        )
        pygame.display.set_caption("Collaborative Multi-Agent Pick and Place")

    def _load_image(self, image_path):
        try:
            return pygame.image.load(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                    f"Error: Image file {image_path} not found."
            )

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


    def _draw_agents_and_objects(self):
        for agent in self.env.agents:
            x, y = agent.position
            cell_center = (
                x * self.env.cell_size + self.env.cell_size // 2,
                y * self.env.cell_size + self.env.cell_size // 2,
            )
            cell_size = self.env.cell_size  # Get the cell size
            icon_size = int(cell_size * 0.8)  # Adjust the agent and object icon size

            # Check if there is an object at the agent's position
            object_at_position = any(obj.position == agent.position for obj in self.env.objects)

            # Check if the agent is a non-picker, not carrying an object, and overlapping with an object
            can_display_side_by_side = (
                not agent.picker
                and agent.carrying_object is None
                and object_at_position
            )

            try:
                if can_display_side_by_side:
                    # Use the non_picker_with_box_icon if conditions are met
                    agent_icon = self.non_picker_with_box_icon
                    icon_size = int(cell_size * 0.8)  # Adjust the icon size
                else:
                    # Select agent icon based on picker status and carrying
                    if agent.picker:
                        if agent.carrying_object is not None:
                            agent_icon = self.picker_carrying_icon
                        else:
                            agent_icon = self.picker_icon
                    else:
                        if agent.carrying_object is not None:
                            agent_icon = self.non_picker_carrying_icon
                        else:
                            agent_icon = self.non_picker_icon

                # Resize agent icon
                agent_icon_resized = pygame.transform.scale(
                    agent_icon, (icon_size, icon_size))

                # Draw agent icon on the offscreen surface
                agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
                self.offscreen_surface.blit(agent_icon_resized, agent_icon_rect)

            except Exception as e:
                # Handle exceptions here (e.g., image loading errors)
                print(f"Error loading icon: {e}")
                # Optionally, you can set a default icon or behavior here


    def _draw_agents(self):
        for agent in self.env.agents:
            x, y = agent.position
            cell_center = (
                x * self.env.cell_size + self.env.cell_size // 2,
                y * self.env.cell_size + self.env.cell_size // 2,
            )
            icon_size = int(self.env.cell_size * 0.95)

            # Select icon based on picker status and carrying
            if agent.picker:
                if agent.carrying_object is not None:
                    agent_icon = self.picker_carrying_icon
                else:
                    agent_icon = self.picker_icon
            else:
                if agent.carrying_object is not None:
                    agent_icon = self.non_picker_carrying_icon
                else:
                    agent_icon = self.non_picker_icon

            # Resize and draw icon
            agent_icon_resized = pygame.transform.scale(
                agent_icon, (icon_size, icon_size))
            agent_icon_rect = agent_icon_resized.get_rect(center=cell_center)
            self.offscreen_surface.blit(agent_icon_resized, agent_icon_rect)

    def _draw_objects(self):
        for obj in self.env.objects:
            x, y = obj.position
            img_size = int(self.env.cell_size * 0.8)
            obj_img_resized = pygame.transform.scale(
                self.object_icon, (img_size, img_size))
            pos = (x * self.env.cell_size + self.env.cell_size // 2 - img_size // 2,
                   y * self.env.cell_size + self.env.cell_size // 2 - img_size // 2)
            self.offscreen_surface.blit(obj_img_resized, pos)

    def _draw_tick_border(self, x, y, color, thickness=8):
        cell_size = self.env.cell_size
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        # Draw the tick border
        pygame.draw.rect(self.offscreen_surface, color, rect, thickness)

    def _draw_goals(self):
        for goal in self.env.goals:
            x, y = goal
            object_at_goal = any(obj.position == (x, y) for obj in self.env.objects)
            dropper_at_goal = any(agent.position == (x, y) and not agent.picker for agent in self.env.agents)
            
            if object_at_goal and (not any(agent.position == (x, y) for agent in self.env.agents) or dropper_at_goal):
                border_color = GREEN
            else:
                border_color = RED
                
            self._draw_tick_border(x, y, border_color)


    def render(self):
        self.offscreen_surface.fill(WHITE)
        self._draw_grid()
        self._draw_goals()
        self._draw_objects()
        # self._draw_agents()
        self._draw_agents_and_objects()
        self.screen.blit(self.offscreen_surface, (0, 0))
        pygame.display.flip()

    def close(self):
        pygame.quit()
