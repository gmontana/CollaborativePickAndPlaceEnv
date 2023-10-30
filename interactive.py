import pygame
from macpp.core.environment import MACPPEnv
from macpp.core.environment import Action


class InteractivePolicy:
    '''
    This class enables the manual control of a two-player game using the keyboard. 

    Attributes:
        actions (list): A list to store the actions chosen for each agent. The default value is [None, None].
        current_agent (int): An integer to track the currently active agent. The default value is 0.
        key_mapping (dict): A dictionary mapping pygame key events to corresponding actions.

    Methods:
        action(): Returns a list of actions for both agents if actions are available, otherwise None.
        handle_key(key): Updates the action list based on the key event received.
    '''

    def __init__(self):
        self.actions = [None, None]
        self.current_agent = 0
        self.key_mapping = {
            pygame.K_UP: Action.UP,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_SPACE: Action.PASS,
            pygame.K_p: Action.WAIT
        }

    def action(self):
        if None not in self.actions:
            actions = self.actions[:]
            self.actions = [None, None]
            return actions
        return None

    def handle_key(self, key):
        if key in self.key_mapping:
            self.actions[self.current_agent] = self.key_mapping[key].value
            self.current_agent = (self.current_agent + 1) % 2


def game_loop(env):
    env.reset()
    done = False
    policy = InteractivePolicy()
    env.render()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                policy.handle_key(event.key)
        actions = policy.action()
        if actions is not None:
            _, reward, done, _ = env.step(actions)
            print(f"Actions: {actions}, Reward: {reward}")
            env.render()
        pygame.time.wait(100)

    env.close()


if __name__ == "__main__":
    env = MACPPEnv(
        grid_size=(5, 5), n_agents=2, n_pickers=1, n_objects=3, debug_mode=True, cell_size=200
    )
    game_loop(env)
