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
PADDLE_SPEED = 5
BALL_INITIAL_SPEED = 4
BALL_MAX_SPEED = 8

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class PongGameAI(BaseGameAI):
    def __init__(self, w=640, h=480):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        pygame.display.set_caption('Pong AI (Single Paddle)')
        
        pygame.display.set_caption('Pong - Close via UI window')
        
        self.clock = pygame.time.Clock()
        self.current_speed = 60

        font_file = resource_path('arial.ttf')
        self.font = pygame.font.Font(font_file, 25)

        self.reset()

    def reset(self):
        # AI controls the right paddle, left paddle is static
        self.ai_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        self.static_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        
        self.ball_x = self.w / 2
        self.ball_y = self.h / 2
        self.ball_vel_x = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        self.ball_vel_y = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        
        self.hits_count = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False
        self.consecutive_hits = 0

    def get_state(self) -> np.ndarray:
        # Ball direction as one-hot: [left, right], [up, down]
        ball_dir_x = [1, 0] if self.ball_vel_x < 0 else [0, 1]
        ball_dir_y = [1, 0] if self.ball_vel_y < 0 else [0, 1]
        paddle_center = self.ai_paddle_y + PADDLE_HEIGHT / 2
        dist_paddle_ball = (self.ball_y - paddle_center) / self.h
        state = [
            self.ai_paddle_y / self.h,           # 1
            self.ball_x / self.w,                # 2
            self.ball_y / self.h,                # 3
            self.ball_vel_x / BALL_MAX_SPEED,    # 4
            self.ball_vel_y / BALL_MAX_SPEED,    # 5
            dist_paddle_ball,                    # 6
            *ball_dir_x,                         # 7, 8
            *ball_dir_y,                         # 9, 10
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = -0.01  # Small negative reward each frame
        done = False
        
        # Action: 0 = up, 1 = down, 2 = stay
        if action_index == 0:
            self.ai_paddle_y -= PADDLE_SPEED
        elif action_index == 1:
            self.ai_paddle_y += PADDLE_SPEED

        # Keep paddle within bounds
        self.ai_paddle_y = max(0, min(self.ai_paddle_y, self.h - PADDLE_HEIGHT))

        # Update ball position
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Ball bounces off top and bottom
        if self.ball_y <= 0 or self.ball_y >= self.h - BALL_SIZE:
            self.ball_vel_y *= -1

        # Ball hits AI paddle (right side)
        if (self.ball_x >= self.w - PADDLE_WIDTH - BALL_SIZE and
            self.ai_paddle_y <= self.ball_y <= self.ai_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1
            self.hits_count += 1
            self.consecutive_hits += 1
            reward = 2.0 + (self.consecutive_hits * 0.1)  # Larger reward for hit and streak
            # Slight speed increase, but capped
            speed_increase = min(1.1, 1.0 + (self.consecutive_hits * 0.02))
            self.ball_vel_x *= speed_increase
            self.ball_vel_y *= speed_increase

        # Ball hits static paddle (left side) - just bounces back
        elif (self.ball_x <= PADDLE_WIDTH and
              self.static_paddle_y <= self.ball_y <= self.static_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1
            self.hits_count += 1
            self.consecutive_hits += 1
            reward = 2.0 + (self.consecutive_hits * 0.1)

        # Ball goes out of bounds
        if self.ball_x < 0 or self.ball_x > self.w - BALL_SIZE:
            done = True
            reward = -2.0  # Smaller penalty for missing
            self.consecutive_hits = 0

        # Game timeout
        if self.frame_iteration > 10000:  # Increased timeout
             done = True
             reward = 5.0  # Positive reward for long games

        self.is_game_over_flag = done

        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.display = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        self.render()
        self.clock.tick(self.current_speed)

        return reward, done, self.hits_count

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 3  # Simplified: up, down, stay

    def get_state_shape(self) -> tuple[int, ...]:
        return (10,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.hits_count

    def render(self) -> None:
        win_w, win_h = self.display.get_size()
        offset_x = (win_w - self.w) // 2 if win_w > self.w else 0
        offset_y = (win_h - self.h) // 2 if win_h > self.h else 0
        self.display.fill(BLACK)
        
        # Draw static paddle (left) in white
        pygame.draw.rect(self.display, WHITE, (0 + offset_x, int(self.static_paddle_y) + offset_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw AI paddle (right) in red
        pygame.draw.rect(self.display, RED, (self.w - PADDLE_WIDTH + offset_x, int(self.ai_paddle_y) + offset_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw ball
        pygame.draw.ellipse(self.display, WHITE, (int(self.ball_x) + offset_x, int(self.ball_y) + offset_y, BALL_SIZE, BALL_SIZE))

        # Draw score
        score_text = self.font.render(f"Hits: {self.hits_count}", True, WHITE)
        self.display.blit(score_text, (self.w / 2 - score_text.get_width() / 2 + offset_x, 20 + offset_y))
        
        # Draw consecutive hits
        if self.consecutive_hits > 0:
            streak_text = self.font.render(f"Streak: {self.consecutive_hits}", True, WHITE)
            self.display.blit(streak_text, (self.w / 2 - streak_text.get_width() / 2 + offset_x, 50 + offset_y))

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

