import pygame
import numpy as np
import random
from game_interface import BaseGameAI

# Galaga constants
WIN_WIDTH = 320
WIN_HEIGHT = 480
PLAYER_WIDTH = 32
PLAYER_HEIGHT = 32
ENEMY_WIDTH = 32
ENEMY_HEIGHT = 32
BULLET_WIDTH = 4
BULLET_HEIGHT = 10
PLAYER_SPEED = 5
ENEMY_SPEED = 2
BULLET_SPEED = 7

class GalagaGameAI(BaseGameAI):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Galaga - Close via UI window')
        self.clock = pygame.time.Clock()
        self.current_speed = 60
        self.font = pygame.font.SysFont('Arial', 18)
        self.reset()

    def reset(self):
        self.player_x = WIN_WIDTH // 2
        self.player_vel = 0
        self.bullet_active = False
        self.bullet_x = 0
        self.bullet_y = 0
        self.enemy_x = random.randint(0, WIN_WIDTH - ENEMY_WIDTH)
        self.enemy_y = 50
        self.score = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False

    def get_state(self) -> np.ndarray:
        # State: [player_x, player_vel, enemy_x, enemy_y, bullet_active, bullet_x, bullet_y, dist_to_enemy_x, dist_to_enemy_y]
        dist_to_enemy_x = (self.enemy_x + ENEMY_WIDTH // 2) - (self.player_x + PLAYER_WIDTH // 2)
        dist_to_enemy_y = self.enemy_y - (WIN_HEIGHT - PLAYER_HEIGHT)
        state = [
            self.player_x / WIN_WIDTH,
            self.player_vel / PLAYER_SPEED,
            self.enemy_x / WIN_WIDTH,
            self.enemy_y / WIN_HEIGHT,
            float(self.bullet_active),
            self.bullet_x / WIN_WIDTH if self.bullet_active else 0.0,
            self.bullet_y / WIN_HEIGHT if self.bullet_active else 0.0,
            dist_to_enemy_x / WIN_WIDTH,
            dist_to_enemy_y / WIN_HEIGHT
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = 0.01  # small reward for staying alive
        done = False

        # Actions: 0=left, 1=right, 2=shoot, 3=do nothing
        if action_index == 0:
            self.player_vel = -PLAYER_SPEED
        elif action_index == 1:
            self.player_vel = PLAYER_SPEED
        elif action_index == 2 and not self.bullet_active:
            self.bullet_active = True
            self.bullet_x = self.player_x + PLAYER_WIDTH // 2 - BULLET_WIDTH // 2
            self.bullet_y = WIN_HEIGHT - PLAYER_HEIGHT
        else:
            self.player_vel = 0

        # Update player position
        self.player_x += self.player_vel
        self.player_x = max(0, min(self.player_x, WIN_WIDTH - PLAYER_WIDTH))

        # Update bullet
        if self.bullet_active:
            self.bullet_y -= BULLET_SPEED
            if self.bullet_y < 0:
                self.bullet_active = False
        
        # Update enemy
        self.enemy_y += ENEMY_SPEED
        if self.enemy_y > WIN_HEIGHT:
            self.enemy_x = random.randint(0, WIN_WIDTH - ENEMY_WIDTH)
            self.enemy_y = 50
            reward = -1.0  # penalty for missing enemy

        # Bullet hits enemy
        if (self.bullet_active and
            self.enemy_x < self.bullet_x < self.enemy_x + ENEMY_WIDTH and
            self.enemy_y < self.bullet_y < self.enemy_y + ENEMY_HEIGHT):
            self.bullet_active = False
            self.enemy_x = random.randint(0, WIN_WIDTH - ENEMY_WIDTH)
            self.enemy_y = 50
            self.score += 1
            reward = 2.0  # reward for hitting enemy

        # Enemy reaches player
        if self.enemy_y + ENEMY_HEIGHT > WIN_HEIGHT - PLAYER_HEIGHT:
            done = True
            reward = -5.0
            self.is_game_over_flag = True

        self.render()
        self.clock.tick(self.current_speed)
        return reward, done, self.score

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 4  # left, right, shoot, do nothing

    def get_state_shape(self) -> tuple[int, ...]:
        return (9,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        self.display.fill((0, 0, 0))
        # Draw player
        pygame.draw.rect(self.display, (0, 255, 255), (self.player_x, WIN_HEIGHT - PLAYER_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT))
        # Draw enemy
        pygame.draw.rect(self.display, (255, 0, 0), (self.enemy_x, self.enemy_y, ENEMY_WIDTH, ENEMY_HEIGHT))
        # Draw bullet
        if self.bullet_active:
            pygame.draw.rect(self.display, (255, 255, 0), (self.bullet_x, self.bullet_y, BULLET_WIDTH, BULLET_HEIGHT))
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Galaga speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.") 