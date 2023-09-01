import pygame 
import collections
import random
import sys

class MovingObject():
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.position = collections.namedtuple("Position", ['x', 'y'])
        self.position.x = random.randrange(DIM_MULT_OF_BLOCK_SIZE)*BLOCK_SIZE
        self.position.y = random.randrange(DIM_MULT_OF_BLOCK_SIZE)*BLOCK_SIZE
        self.dimension = collections.namedtuple("Dimension", ['x', 'y'])
        self.dimension.x = BLOCK_SIZE
        self.dimension.y = BLOCK_SIZE
    def move(self, key, position):
        if key=='LEFT' and self.position.x>0: 
            self.position.x -= BLOCK_SIZE
        if key=='RIGHT' and self.position.x<GRID_WIDTH-BLOCK_SIZE:
            self.position.x += BLOCK_SIZE
        if key=='UP' and self.position.y>0: 
            self.position.y -= BLOCK_SIZE
        if key=="DOWN"  and self.position.y<GRID_HEIGHT-BLOCK_SIZE:
            self.position.y += BLOCK_SIZE

def InitObjects():
    objects=[]
    for i in range(NO_OBJECTS):
        print("Adding object ", i)
        objects.append(MovingObject(NAMES[i], COLORS[i]))
    return objects

def GenerateGraph(objects):
    graph = [ [ [0] * EDGE_SIZE for i in range(NO_OBJECTS) ] for j in range(NO_OBJECTS) ]
    for i in range(NO_OBJECTS):
        for j in range(NO_OBJECTS):
            edge = [0]*EDGE_SIZE # initialise edge weights with zeros 
            obj1 = objects[i]
            obj2 = objects[j]
            if obj1.position.y > obj2.position.y: # above/below
                edge[0] = 1
            if obj1.position.x > obj2.position.x: # left/right
                edge[1] = 1
            # if obj1.position.x == obj2.position.x: # horizontal align
            #     edge[2] = 1
            # if obj1.position.y == obj2.position.y: # vertical align
            #     edge[3] = 1
            graph[i][j]=edge
    return graph

def PrintGraph(objects, graph):
    for i in range(0, NO_OBJECTS):
        for j in range(i+1, NO_OBJECTS):
            print('  {} is BELOW {}: {}'.format(objects[i].name, objects[j].name, bool(graph[i][j][0])))
            print('  {} is RIGHT {}: {}'.format(objects[i].name, objects[j].name, bool(graph[i][j][1])))

def TargetGraph():
    graph = [ [ [0] * EDGE_SIZE for i in range(NO_OBJECTS) ] for j in range(NO_OBJECTS) ]
    for i in range(0, NO_OBJECTS):
        for j in range(i+1, NO_OBJECTS):
            for z in range(EDGE_SIZE):
                graph[i][j][z] = random.randrange(2)
    return graph

def GraphReward(graph1, graph2):
    rewards=[]
    for i in range(0, NO_OBJECTS):
        for j in range(i+1, NO_OBJECTS):
            rewards.append(EdgeReward(graph1[i][j], graph2[i][j]))
    return sum(rewards)/len(rewards)

def EdgeReward(edge1, edge2):
    count=0
    for i in range(len(edge1)):
        if edge1[i]==edge2[i]:
            count+=1
    return float(count)/len(edge1)

# Define RGB colors for objects
RED = (255, 0,0)
YELLOW = (255,255,0)
GREEN = (128, 255,0)
BLUE = (0,0,255)
WHITE= (255,255,255)
GRAY= (224,224,224)
PINK= (255, 204, 204)
COLORS=[PINK, YELLOW, GREEN, BLUE]
NAMES=['Pink', 'Yellow', 'Green', 'Blue']

# Grid size
BLOCK_SIZE = 20 # size of each block
DIM_MULT_OF_BLOCK_SIZE=15 # number of blocks in each direction
GRID_WIDTH=BLOCK_SIZE*DIM_MULT_OF_BLOCK_SIZE
GRID_HEIGHT=BLOCK_SIZE*DIM_MULT_OF_BLOCK_SIZE

# Number of objects
NO_OBJECTS = 3

# Edge size (number of binary relationships)
EDGE_SIZE = 2

# Generate objects for the target graph and target graph
target_objects = InitObjects()
target_graph = GenerateGraph(target_objects)

# Generate working objects
objects = InitObjects()

# Initialise pygame and display
pygame.init() 
# win = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT)) 
win = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT)) 
pygame.display.set_caption("Relational learning") 
  
  
# Indicates pygame is running 
run = True

# Active object
activeobject = 0

# Infinite loop  
while run: 

    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            # The fun ends
            run = False

    # Listen to key presses
    keys = pygame.key.get_pressed()

    # Select the active object 
    if keys[pygame.K_1]:
        activeobject = 0

    if keys[pygame.K_2]:
        activeobject = 1

    if keys[pygame.K_3]:
        activeobject = 2

    # Reset the target graph
    if keys[pygame.K_SPACE]:
        target_objects = InitObjects()
        target_graph = GenerateGraph(target_objects)

    # Arrow keys move the active object 
    if keys[pygame.K_LEFT]:
        objects[activeobject].move('LEFT', objects[activeobject].position)
          
    if keys[pygame.K_RIGHT]:
        objects[activeobject].move('RIGHT', objects[activeobject].position)
         
    if keys[pygame.K_UP]:
        objects[activeobject].move('UP', objects[activeobject].position)
          
    if keys[pygame.K_DOWN]:
        objects[activeobject].move('DOWN', objects[activeobject].position)

    # # Update the graph
    # for i in range(NO_OBJECTS):
    #     for j in range(NO_OBJECTS):

    #         edge=GetEdge(objects[i], objects[j])
    #         print(i, j, edge)
              
    # Draw the grid
    win.fill((0, 0, 0))  # Black background 
    for i in range(DIM_MULT_OF_BLOCK_SIZE+1):
        pygame.draw.line(win, GRAY, (0,i*BLOCK_SIZE), (GRID_WIDTH,i*BLOCK_SIZE)) # Horizontal line
        pygame.draw.line(win, GRAY, (i*BLOCK_SIZE,0), (i*BLOCK_SIZE,GRID_HEIGHT))  # Vertical line
      
    # Highlight the active object
    pygame.draw.rect(win, RED, (objects[activeobject].position.x-2, objects[activeobject].position.y-2, objects[activeobject].dimension.x+4, objects[activeobject].dimension.y+6))

    # Draw all the objects on screen 
    for i in range(NO_OBJECTS):
        pygame.draw.rect(win, objects[i].color, (objects[i].position.x, objects[i].position.y, objects[i].dimension.x, objects[i].dimension.y))
      
    # Update the current graph
    current_graph = GenerateGraph(objects)

    # Print the current graph
    print('Current graph:')
    PrintGraph(objects, current_graph)

    # Print the target graph
    print('Target graph:')
    PrintGraph(target_objects, target_graph)

    # Calculate the reward
    reward=round(GraphReward(current_graph, target_graph),3)
    print('Reward: {}'.format(reward))

    # Display the reward 
    text=str(reward)
    font = pygame.font.Font('freesansbold.ttf', 18)
    textSurface = font.render(text, True, YELLOW)
    textRect= textSurface.get_rect()
    textRect.center=(50,20)
    win.blit(textSurface, textRect)

    # Refresh the window and delay
    pygame.display.update()  
    pygame.time.delay(80) 
  
pygame.quit() 