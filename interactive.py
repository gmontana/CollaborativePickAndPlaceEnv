import pygame
from macpp.core.environment import MultiAgentPickAndPlace


class InteractivePolicy:
    def __init__(self, env):
        self.env = env
        self.move_a = [False for _ in range(5)]
        self.move_b = [False for _ in range(5)]

        if env.renderer:
            env.renderer.window.on_key_press = self.key_press
            env.renderer.window.on_key_release = self.key_release

    def action(self, obs):
        u = 4  # Default to "pass"
        if self.move_a[0]:
            u = 0  # move_up
        if self.move_a[1]:
            u = 1  # move_down
        if self.move_a[2]:
            u = 2  # move_left
        if self.move_a[3]:
            u = 3  # move_right

        v = 4  # Default to "pass"
        if self.move_b[0]:
            v = 0  # move_up
        if self.move_b[1]:
            v = 1  # move_down
        if self.move_b[2]:
            v = 2  # move_left
        if self.move_b[3]:
            v = 3  # move_right

        return [u, v]

    def key_press(self, k, mod):

        # print(f'Key pressed: {k}')
        if k == pygame.K_k:
            self.move_a[0] = True
        if k == pygame.K_j:
            self.move_a[1] = True
        if k == pygame.K_j:
            self.move_a[2] = True
        if k == pygame.K_l:
            self.move_a[3] = True
        if k == pygame.K_SPACE:
            self.move_a[4] = True

        if k == pygame.K_s:
            self.move_b[0] = True
        if k == pygame.K_d:
            self.move_b[1] = True
        if k == pygame.K_a:
            self.move_b[2] = True
        if k == pygame.K_f:
            self.move_b[3] = True
        if k == pygame.K_x:
            self.move_b[4] = True

    def key_release(self, k, mod):

        # print(f'Key released: {k}')
        if k == pygame.K_UP:
            self.move_a[0] = False
        if k == pygame.K_DOWN:
            self.move_a[1] = False
        if k == pygame.K_LEFT:
            self.move_a[2] = False
        if k == pygame.K_RIGHT:
            self.move_a[3] = False
        if k == pygame.K_SPACE:
            self.move_a[4] = False

        if k == pygame.K_w:
            self.move_b[0] = False
        if k == pygame.K_s:
            self.move_b[1] = False
        if k == pygame.K_a:
            self.move_b[2] = False
        if k == pygame.K_d:
            self.move_b[3] = False
        if k == pygame.K_x:
            self.move_b[4] = False


def game_loop(env):
    obs = env.reset()
    done = False
    policy = InteractivePolicy(env)

    while not done:
        actions = policy.action(obs)
        nobs, nreward, ndone, _ = env.step(actions)
        env.render()
        pygame.time.wait(100)

    env.close()

if __name__ == "__main__":
    env = MultiAgentPickAndPlace(
        width=3, length=3, n_agents=2, n_pickers=1
    )
    game_loop(env)
