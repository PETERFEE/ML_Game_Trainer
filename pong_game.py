# pong_game.py
import pygame
import random
import numpy as np

from game_interface import BaseGameAI
from utils import resource_path

# Constants for Pong
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 80
BALL_SIZE = 10
PADDLE_SPEED = 7 # Speed at which the AI's paddle can move
BALL_INITIAL_SPEED = 5 # Initial speed of the ball

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class PongGameAI(BaseGameAI):
    def __init__(self, w=640, h=480):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Pong AI (Two Paddles)')
        self.clock = pygame.time.Clock()
        self.current_speed = 60 # FPS for Pong (can be higher for AI)

        font_file = resource_path('arial.ttf')
        self.font = pygame.font.Font(font_file, 25)

        self.reset()

    def reset(self):
        # Initialize both paddles to the center
        self.left_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        self.right_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        
        # Initialize ball position and random direction
        self.ball_x = self.w / 2
        self.ball_y = self.h / 2
        self.ball_vel_x = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        self.ball_vel_y = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        
        self.hits_count = 0 # Single score: number of times ball was hit by either paddle
        self.frame_iteration = 0
        self.is_game_over_flag = False

    def get_state(self) -> np.ndarray:
        # State representation for a single AI controlling both paddles:
        # 0: self.left_paddle_y / self.h (Normalized left paddle Y)
        # 1: self.right_paddle_y / self.h (Normalized right paddle Y)
        # 2: self.ball_x / self.w (Normalized ball X)
        # 3: self.ball_y / self.h (Normalized ball Y)
        # 4: self.ball_vel_x / BALL_INITIAL_SPEED (Normalized ball velocity X)
        # 5: self.ball_vel_y / BALL_INITIAL_SPEED (Normalized ball velocity Y)
        # 6: (self.ball_y - (self.left_paddle_y + PADDLE_HEIGHT/2)) / self.h (Relative ball Y to left paddle center)
        # 7: (self.ball_y - (self.right_paddle_y + PADDLE_HEIGHT/2)) / self.h (Relative ball Y to right paddle center)

        state = [
            self.left_paddle_y / self.h,
            self.right_paddle_y / self.h,
            self.ball_x / self.w,
            self.ball_y / self.h,
            self.ball_vel_x / BALL_INITIAL_SPEED,
            self.ball_vel_y / BALL_INITIAL_SPEED,
            (self.ball_y - (self.left_paddle_y + PADDLE_HEIGHT/2)) / self.h,
            (self.ball_y - (self.right_paddle_y + PADDLE_HEIGHT/2)) / self.h
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = 0
        done = False
        
        # Decode action_index for two paddles:
        # Action space (0-8):
        # 0: L_UP, R_UP
        # 1: L_UP, R_DOWN
        # 2: L_UP, R_STAY
        # 3: L_DOWN, R_UP
        # 4: L_DOWN, R_DOWN
        # 5: L_DOWN, R_STAY
        # 6: L_STAY, R_UP
        # 7: L_STAY, R_DOWN
        # 8: L_STAY, R_STAY

        left_move = action_index // 3   # 0, 1, 2 for UP, DOWN, STAY
        right_move = action_index % 3   # 0, 1, 2 for UP, DOWN, STAY

        # Apply movement for left paddle
        if left_move == 0: # UP
            self.left_paddle_y -= PADDLE_SPEED
        elif left_move == 1: # DOWN
            self.left_paddle_y += PADDLE_SPEED
        # else: STAY (left_move == 2)

        # Apply movement for right paddle
        if right_move == 0: # UP
            self.right_paddle_y -= PADDLE_SPEED
        elif right_move == 1: # DOWN
            self.right_paddle_y += PADDLE_SPEED
        # else: STAY (right_move == 2)

        # Keep paddles within bounds
        self.left_paddle_y = max(0, min(self.left_paddle_y, self.h - PADDLE_HEIGHT))
        self.right_paddle_y = max(0, min(self.right_paddle_y, self.h - PADDLE_HEIGHT))

        # Update ball position
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Ball collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.h - BALL_SIZE:
            self.ball_vel_y *= -1
            # reward -= 0.1 # Small penalty for hitting top/bottom (optional)

        # Ball collision with paddles
        # Left paddle collision
        if (self.ball_x <= PADDLE_WIDTH and
            self.left_paddle_y <= self.ball_y <= self.left_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1.05 # Increase speed slightly on hit
            self.hits_count += 1
            reward = 1.0 # Reward for hitting ball

        # Right paddle collision
        elif (self.ball_x >= self.w - PADDLE_WIDTH - BALL_SIZE and
            self.right_paddle_y <= self.ball_y <= self.right_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1.05 # Increase speed slightly on hit
            self.hits_count += 1
            reward = 1.0 # Reward for hitting ball

        # Check for game over (ball went past either paddle)
        if self.ball_x < 0 or self.ball_x > self.w - BALL_SIZE:
            done = True
            reward = -10.0 # Large penalty for letting ball pass

        # Timeout for long games (optional, prevents infinite loops)
        if self.frame_iteration > 5000: # Max frames per round
             done = True
             reward = -1.0 # Small penalty for timeout if no score

        self.is_game_over_flag = done # Update internal flag

        self.render()
        self.clock.tick(self.current_speed)

        return reward, done, self.hits_count

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 9 # 3 actions for left paddle * 3 actions for right paddle

    def get_state_shape(self) -> tuple[int, ...]:
        return (8,) # Based on the 8 features in get_state()

    def get_state_is_image(self) -> bool:
        return False # This game uses a feature vector, not an image

    def get_score(self) -> int:
        return self.hits_count

    def render(self) -> None:
        self.display.fill(BLACK)
        
        # Draw left paddle
        pygame.draw.rect(self.display, WHITE, (0, self.left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        # Draw right paddle
        pygame.draw.rect(self.display, WHITE, (self.w - PADDLE_WIDTH, self.right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.ellipse(self.display, WHITE, (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))

        # Draw hits count (single score)
        score_text = self.font.render(f"Hits: {self.hits_count}", True, WHITE)
        self.display.blit(score_text, (self.w / 2 - score_text.get_width() / 2, 20))

        pygame.display.flip()
    
    def close(self) -> None:
        pygame.quit()

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Pong speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.")

