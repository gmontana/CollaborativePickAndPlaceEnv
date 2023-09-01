import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Space Invaders")

# Set up the game clock
clock = pygame.time.Clock()

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up fonts
font = pygame.font.Font(None, 32)

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT]:
            self.rect.x += 5
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.x_speed = 5

    def update(self):
        self.rect.x += self.x_speed
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.x_speed *= -1
            self.rect.y += 50

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 20))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.y_speed = -10

    def update(self):
        self.rect.y += self.y_speed

class Game:
    def __init__(self):
        self.all_sprites = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.player = Player(SCREEN_WIDTH//2, SCREEN_HEIGHT-60)
        self.all_sprites.add(self.player)
        self.enemy_timer = 0
        self.score = 0
        self.text_x = 10
        self.text_y = 10
        self.score_text = font.render("Score: " + str(self.score), True, WHITE)

    def update(self, keys):
        # Update sprites
        self.all_sprites.update(keys)
        self.enemies.update()
        self.bullets.update()

        # Spawn new enemies
        self.enemy_timer += 1
        if self.enemy_timer >= 60:
            enemy = Enemy(random.randint(50, SCREEN_WIDTH-50), 0)
            self.all_sprites.add(enemy)
            self.enemies
            self.enemy_timer = 0

        # Check for collisions
        for bullet in self.bullets:
            enemy_hit = pygame.sprite.spritecollide(bullet, self.enemies, True)
            if enemy_hit:
                self.score += 1
                self.score_text = font.render("Score: " + str(self.score), True, WHITE)
                bullet.kill()
        player_hit = pygame.sprite.spritecollide(self.player, self.enemies, False)
        if player_hit:
            self.game_over()

        # Draw sprites
        screen.fill(BLACK)
        self.all_sprites.draw(screen)

        # Draw score
        screen.blit(self.score_text, (self.text_x, self.text_y))

        # Update the display
        pygame.display.update()

        # Set the game clock
        clock.tick(60)

    def fire_bullet(self):
        bullet = Bullet(self.player.rect.centerx, self.player.rect.top)
        self.all_sprites.add(bullet)
        self.bullets.add(bullet)

    def game_over(self):
        game_over_text = font.render("Game Over!", True, WHITE)
        screen.blit(game_over_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
        pygame.display.update()
        pygame.time.wait(2000)
        pygame.quit()
        quit()

def main():
    game = Game()

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.fire_bullet()

        # Get key input
        keys = pygame.key.get_pressed()

        # Update the game
        game.update(keys)
