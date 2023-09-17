import pygame
from macpp.core.environment import MultiAgentPickAndPlace
from macpp.core.environment import Action

class InteractivePolicy:
    '''
    Class to manually control a two-player game using the keyboard 
    '''
    def __init__(self):
        self.move_a = {Action.UP: False, Action.DOWN: False, Action.LEFT: False, Action.RIGHT: False, Action.PASS: False}
        self.move_b = {Action.UP: False, Action.DOWN: False, Action.LEFT: False, Action.RIGHT: False, Action.PASS: False}

    def action(self):
        actions = [
            next((action for action, pressed in self.move_a.items() if pressed), None),
            next((action for action, pressed in self.move_b.items() if pressed), None)
        ]
        return actions

    def handle_key(self, key, pressed):
        key_mappings_a = {
            pygame.K_UP: Action.UP,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_SPACE: Action.PASS
        }

        key_mappings_b = {
            pygame.K_y: Action.UP,
            pygame.K_b: Action.DOWN,
            pygame.K_g: Action.LEFT,
            pygame.K_h: Action.RIGHT,
            pygame.K_x: Action.PASS
        }

        if key in key_mappings_a:
            self.move_a[key_mappings_a[key]] = pressed
        elif key in key_mappings_b:
            self.move_b[key_mappings_b[key]] = pressed

def game_loop(env):
    env.reset()
    done = False
    policy = InteractivePolicy()
    env.render()

    while not done:
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:
                done = True
            if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
                pressed = event.type == pygame.KEYDOWN
                policy.handle_key(event.key, pressed)

        actions = policy.action()
        if actions and all(action is not None for action in actions):
            _, _, done, _ = env.step(actions)
            env.render()
        pygame.time.wait(100)

    env.close()

if __name__ == "__main__":
    env = MultiAgentPickAndPlace(
        width=4, length=4, n_agents=2, n_pickers=1, n_objects=2, debug_mode=True
    )
    game_loop(env)

