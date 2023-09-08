import pygame
import random

# define constants
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_SPEED = 1
BALL_SIZE = 10
BALL_SPEED = 0.1
FONT_SIZE = 50

class Paddle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move_up(self):
        self.y -= PADDLE_SPEED
    
    def move_down(self):
        self.y += PADDLE_SPEED
    
    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT))

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed_x = BALL_SPEED * random.choice((-1, 1))
        self.speed_y = BALL_SPEED * random.choice((-1, 1))
    
    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
    
    def bounce_off_walls(self):
        if self.y < 0 or self.y > DISPLAY_HEIGHT - BALL_SIZE:
            self.speed_y *= -1
    
    def bounce_off_paddles(self, left_paddle, right_paddle):
        if self.x < left_paddle.x + PADDLE_WIDTH and left_paddle.y < self.y < left_paddle.y + PADDLE_HEIGHT:
            self.speed_x *= -1
        elif self.x > right_paddle.x and right_paddle.y < self.y < right_paddle.y + PADDLE_HEIGHT:
            self.speed_x *= -1
    
    def reset(self):
        self.x = DISPLAY_WIDTH // 2 - BALL_SIZE // 2
        self.y = DISPLAY_HEIGHT // 2 - BALL_SIZE // 2
        self.speed_x = BALL_SPEED * random.choice((-1, 1))
        self.speed_y = BALL_SPEED * random.choice((-1, 1))

    def check_out_of_bounds(self):
        if self.x < 0:
            return 'right'
        elif self.x > DISPLAY_WIDTH - BALL_SIZE:
            return 'left'
        else:
            return None
    
    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(self.x, self.y, BALL_SIZE, BALL_SIZE))

class Scoreboard:
    def __init__(self):
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.left_score = 0
        self.right_score = 0
    
    def increase_score(self, side):
        if side == 'left':
            self.left_score += 1
        elif side == 'right':
            self.right_score += 1
    
    def draw(self, surface):
        left_text = self.font.render(str(self.left_score), True, (255, 255, 255))
        right_text = self.font.render(str(self.right_score), True, (255, 255, 255))
        surface.blit(left_text, (DISPLAY_WIDTH // 4 - FONT_SIZE // 4, FONT_SIZE // 4))
        surface.blit(right_text, (3 * DISPLAY_WIDTH // 4 - FONT_SIZE // 4, FONT_SIZE // 4))

# initialize Pygame
pygame.init()

# set up the display
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('Ping Pong')

# create game objects
left_paddle = Paddle(50, DISPLAY_HEIGHT // 2 - PADDLE_HEIGHT // 2)
right_paddle = Paddle(DISPLAY_WIDTH - 50 - PADDLE_WIDTH, DISPLAY_HEIGHT // 2 - PADDLE_HEIGHT // 2)
ball = Ball(DISPLAY_WIDTH // 2 - BALL_SIZE // 2, DISPLAY_HEIGHT // 2 - BALL_SIZE // 2)
scoreboard = Scoreboard()

# game loop
running = True
while running:
    # event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # move the paddles
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        left_paddle.move_up()
    if keys[pygame.K_s]:
        left_paddle.move_down()
    if keys[pygame.K_UP]:
        right_paddle.move_up()
    if keys[pygame.K_DOWN]:
        right_paddle.move_down()

    # keep the paddles on the screen
    if left_paddle.y < 0:
        left_paddle.y = 0
    elif left_paddle.y > DISPLAY_HEIGHT - PADDLE_HEIGHT:
        left_paddle.y = DISPLAY_HEIGHT - PADDLE_HEIGHT

    if right_paddle.y < 0:
        right_paddle.y = 0
    elif right_paddle.y > DISPLAY_HEIGHT - PADDLE_HEIGHT:
        right_paddle.y = DISPLAY_HEIGHT - PADDLE_HEIGHT

    # move the ball
    ball.move()

    # bounce the ball off the walls and paddles
    ball.bounce_off_walls()
    ball.bounce_off_paddles(left_paddle, right_paddle)

    # check if the ball went out of bounds
    out_of_bounds = ball.check_out_of_bounds()
    if out_of_bounds is not None:
        scoreboard.increase_score(out_of_bounds)
        ball.reset()

    # draw game objects
    display.fill((0, 0, 0))
    left_paddle.draw(display)
    right_paddle.draw(display)
    ball.draw(display)
    scoreboard.draw(display)

    # update the display
    pygame.display.update()

# quit Pygame
pygame.quit()

