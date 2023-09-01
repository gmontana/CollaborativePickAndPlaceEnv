import pygame
import sys
import random
import math
import numpy as np

SCREEN_SIZE = WIDTH, HEIGHT = (800, 800)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
GRAY = (99, 102,106)
RADIUS = 20
N_STONES = 20
STEP_SIZE = 20
TIME_OUT = 1000

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Stones')
fps = pygame.time.Clock()
paused = False

class Stone:
    def __init__(self, x, y, MOVE):
        self.x = x
        self.y = y
        self.MOVE = MOVE

def render():
    screen.fill(BLACK)
    for i in range(N_STONES):
        if stones[i].MOVE:
            COLOR=GREEN
            TEXT_COLOR=BLACK
        else:
            COLOR=GRAY
            TEXT_COLOR=WHITE
        pygame.draw.circle(screen, COLOR, (stones[i].x, stones[i].y), RADIUS)
        text  = font.render(str(i+1), True, TEXT_COLOR)
        screen.blit(text,(stones[i].x-5,stones[i].y-5))
    pygame.display.update()
    fps.tick(60)

def check_collision(stone1, stone2):
    dx = stone2.x - stone1.x
    dy = stone2.y - stone1.y
    d = math.sqrt( ( dx*dx + dy*dy) )
    if (d <= 2*RADIUS):
        return True
    else:
        return False

def init_stones():
    stones = []
    locations = np.linspace(1, N_STONES)
    for i in range(N_STONES):
        stones.append(Stone(random.randint(1,WIDTH), random.randint(1,HEIGHT), True))
        #stones.append(Stone((i+1)*RADIUS*5, (i+1)*RADIUS*3, True))
    return stones

def update_stones():
    for i in range(N_STONES):
        if stones[i].MOVE is True:
            # update x coord
            xadd = random.randint(-STEP_SIZE,STEP_SIZE)
            if stones[i].x + xadd > 1 and  stones[i].x + xadd < WIDTH:
                stones[i].x += xadd  
            # update y coord
            yadd = random.randint(-STEP_SIZE,STEP_SIZE)
            if stones[i].y + yadd > 1 and  stones[i].y + yadd < HEIGHT:
                stones[i].y += yadd  
            # check for collision with any other stone
            indeces = list(range(N_STONES))
            for j in (indeces[:i] + indeces[i+ 1:]):
                if (check_collision(stones[i],stones[j]) is True):
                    stones[i].MOVE=False
                    stones[j].MOVE=False
                    adj_matrix[j][i]= adj_matrix[i][j]=1
               
stones = init_stones()
adj_matrix = np.zeros((N_STONES, N_STONES), int)
time=1
font = pygame.font.SysFont("Arial", 15)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                paused = not paused
    if not paused:
        update_stones()
        render()
        print("t=", time)
        print(adj_matrix)
        if not True in [s.MOVE for s in stones] or time == TIME_OUT: # end episode
            pygame.image.save(screen, "goal.jpg")
            break
        time +=1 
