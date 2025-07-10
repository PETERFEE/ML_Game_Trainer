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
PADDLE_SPEED = 7
BALL_INITIAL_SPEED = 5

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class PongGameAI(BaseGameAI):
    def __init__(self, w=640, h=480):
        pygame.init()
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Pong AI (Two Paddles)')
        
        pygame.display.set_caption('Pong - Close via UI window')
        
        self.clock = pygame.time.Clock()
        self.current_speed = 60

        font_file = resource_path('arial.ttf')
        self.font = pygame.font.Font(font_file, 25)

        self.reset()

    def reset(self):
        self.left_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        self.right_paddle_y = self.h / 2 - PADDLE_HEIGHT / 2
        
        self.ball_x = self.w / 2
        self.ball_y = self.h / 2
        self.ball_vel_x = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        self.ball_vel_y = random.choice([-1, 1]) * BALL_INITIAL_SPEED
        
        self.hits_count = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False

    def get_state(self) -> np.ndarray:
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
        
        left_move = action_index // 3
        right_move = action_index % 3

        if left_move == 0:
            self.left_paddle_y -= PADDLE_SPEED
        elif left_move == 1:
            self.left_paddle_y += PADDLE_SPEED

        if right_move == 0:
            self.right_paddle_y -= PADDLE_SPEED
        elif right_move == 1:
            self.right_paddle_y += PADDLE_SPEED

        self.left_paddle_y = max(0, min(self.left_paddle_y, self.h - PADDLE_HEIGHT))
        self.right_paddle_y = max(0, min(self.right_paddle_y, self.h - PADDLE_HEIGHT))

        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        if self.ball_y <= 0 or self.ball_y >= self.h - BALL_SIZE:
            self.ball_vel_y *= -1

        if (self.ball_x <= PADDLE_WIDTH and
            self.left_paddle_y <= self.ball_y <= self.left_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1.05
            self.hits_count += 1
            reward = 1.0

        elif (self.ball_x >= self.w - PADDLE_WIDTH - BALL_SIZE and
            self.right_paddle_y <= self.ball_y <= self.right_paddle_y + PADDLE_HEIGHT):
            self.ball_vel_x *= -1.05
            self.hits_count += 1
            reward = 1.0

        if self.ball_x < 0 or self.ball_x > self.w - BALL_SIZE:
            done = True
            reward = -10.0

        if self.frame_iteration > 5000:
             done = True
             reward = -1.0

        self.is_game_over_flag = done

        self.render()
        self.clock.tick(self.current_speed)

        return reward, done, self.hits_count

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 9

    def get_state_shape(self) -> tuple[int, ...]:
        return (8,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.hits_count

    def render(self) -> None:
        self.display.fill(BLACK)
        
        pygame.draw.rect(self.display, WHITE, (0, int(self.left_paddle_y), PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.display, WHITE, (self.w - PADDLE_WIDTH, int(self.right_paddle_y), PADDLE_WIDTH, PADDLE_HEIGHT))
        
        pygame.draw.ellipse(self.display, WHITE, (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE))

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

