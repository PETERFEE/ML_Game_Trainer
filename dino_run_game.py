import pygame
import numpy as np
import random
from game_interface import BaseGameAI

# Dino Run constants
WIN_WIDTH = 600
WIN_HEIGHT = 200
DINO_WIDTH = 40
DINO_HEIGHT = 40
GROUND_Y = WIN_HEIGHT - DINO_HEIGHT - 10
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 40
OBSTACLE_SPEED_MIN = 5
OBSTACLE_SPEED_MAX = 10
OBSTACLE_GAP_MIN = 0
OBSTACLE_GAP_MAX = 150
JUMP_VEL = -12
GRAVITY = 1

class DinoRunGameAI(BaseGameAI):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Dino Run - Close via UI window')
        self.clock = pygame.time.Clock()
        self.current_speed = 60
        self.font = pygame.font.SysFont('Arial', 18)
        self.reset()

    def reset(self):
        self.dino_y = GROUND_Y
        self.dino_vel = 0
        self.on_ground = True
        self.obstacle_x = WIN_WIDTH
        self.obstacle_y = GROUND_Y
        self.obstacle_speed = random.randint(OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX)
        self.next_obstacle_gap = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)
        self.score = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False

    def get_state(self) -> np.ndarray:
        # State: [dino_y, dino_vel, dist_to_obstacle, obstacle_width, obstacle_height, on_ground]
        dist_to_obstacle = (self.obstacle_x - 50) / WIN_WIDTH  # Dino x is fixed at 50
        state = [
            self.dino_y / WIN_HEIGHT,
            self.dino_vel / 20.0,
            dist_to_obstacle,
            OBSTACLE_WIDTH / WIN_WIDTH,
            OBSTACLE_HEIGHT / WIN_HEIGHT,
            float(self.on_ground)
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = 0.1  # small reward for staying alive
        done = False

        # Action: 0 = do nothing, 1 = jump
        if action_index == 1 and self.on_ground:
            self.dino_vel = JUMP_VEL
            self.on_ground = False

        # Update dino position
        self.dino_y += self.dino_vel
        if not self.on_ground:
            self.dino_vel += GRAVITY
        if self.dino_y >= GROUND_Y:
            self.dino_y = GROUND_Y
            self.dino_vel = 0
            self.on_ground = True

        # Move obstacle
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < -OBSTACLE_WIDTH:
            self.next_obstacle_gap = random.randint(OBSTACLE_GAP_MIN, OBSTACLE_GAP_MAX)
            self.obstacle_x = WIN_WIDTH + self.next_obstacle_gap
            self.obstacle_speed = random.randint(OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX)
            self.score += 1
            reward = 1.0  # reward for passing obstacle

        # Collision detection
        dino_rect = pygame.Rect(50, int(self.dino_y), DINO_WIDTH, DINO_HEIGHT)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
        if dino_rect.colliderect(obstacle_rect):
            done = True
            reward = -5.0
            self.is_game_over_flag = True

        self.render()
        self.clock.tick(self.current_speed)
        return reward, done, self.score

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 2  # do nothing, jump

    def get_state_shape(self) -> tuple[int, ...]:
        return (6,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        self.display.fill((235, 235, 235))  # Light gray background
        # Draw ground
        pygame.draw.rect(self.display, (100, 100, 100), (0, GROUND_Y + DINO_HEIGHT, WIN_WIDTH, 10))
        # Draw dino
        pygame.draw.rect(self.display, (0, 0, 0), (50, int(self.dino_y), DINO_WIDTH, DINO_HEIGHT))
        # Draw obstacle
        pygame.draw.rect(self.display, (34, 139, 34), (self.obstacle_x, self.obstacle_y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.display.blit(score_text, (10, 10))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Dino Run speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.")
