import pygame
import numpy as np
import random
from game_interface import BaseGameAI

# Flappy Bird constants
WIN_WIDTH = 288
WIN_HEIGHT = 512
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 100
GRAVITY = 0.5
FLAP_STRENGTH = -7.5

class FlappyBirdGameAI(BaseGameAI):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption('Flappy Bird - Close via UI window')
        self.clock = pygame.time.Clock()
        self.current_speed = 60
        self.font = pygame.font.SysFont('Arial', 18)
        self.reset()

    def reset(self):
        self.bird_y = WIN_HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = WIN_WIDTH
        self.pipe_gap_y = random.randint(80, WIN_HEIGHT - 80 - PIPE_GAP)
        self.score = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False

    def get_state(self) -> np.ndarray:
        # State: [bird_y, bird_vel, dist_to_pipe, pipe_gap_y, bird_y - pipe_gap_y, bird_y - (pipe_gap_y+PIPE_GAP)]
        dist_to_pipe = self.pipe_x - 50  # Bird x is fixed at 50
        state = [
            self.bird_y / WIN_HEIGHT,
            self.bird_vel / 10.0,
            dist_to_pipe / WIN_WIDTH,
            self.pipe_gap_y / WIN_HEIGHT,
            (self.bird_y - self.pipe_gap_y) / WIN_HEIGHT,
            (self.bird_y - (self.pipe_gap_y + PIPE_GAP)) / WIN_HEIGHT
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = 0.1  # small reward for staying alive
        done = False

        # Action: 0 = do nothing, 1 = flap
        if action_index == 1:
            self.bird_vel = FLAP_STRENGTH
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel

        # Move pipe
        self.pipe_x -= 3
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIN_WIDTH
            self.pipe_gap_y = random.randint(80, WIN_HEIGHT - 80 - PIPE_GAP)
            self.score += 1
            reward = 1.0  # reward for passing a pipe

        # Collision detection
        if (
            self.bird_y < 0 or self.bird_y + BIRD_HEIGHT > WIN_HEIGHT or
            (self.pipe_x < 50 + BIRD_WIDTH < self.pipe_x + PIPE_WIDTH and
             (self.bird_y < self.pipe_gap_y or self.bird_y + BIRD_HEIGHT > self.pipe_gap_y + PIPE_GAP))
        ):
            done = True
            reward = -5.0
            self.is_game_over_flag = True

        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.display = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        self.render()
        self.clock.tick(self.current_speed)
        return reward, done, self.score

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 2  # 0 = do nothing, 1 = flap

    def get_state_shape(self) -> tuple[int, ...]:
        return (6,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        win_w, win_h = self.display.get_size()
        offset_x = (win_w - WIN_WIDTH) // 2 if win_w > WIN_WIDTH else 0
        offset_y = (win_h - WIN_HEIGHT) // 2 if win_h > WIN_HEIGHT else 0
        self.display.fill((135, 206, 235))  # Sky blue
        # Draw bird
        pygame.draw.rect(self.display, (255, 255, 0), (50+offset_x, int(self.bird_y)+offset_y, BIRD_WIDTH, BIRD_HEIGHT))
        # Draw pipe
        pygame.draw.rect(self.display, (0, 255, 0), (self.pipe_x+offset_x, 0+offset_y, PIPE_WIDTH, self.pipe_gap_y))
        pygame.draw.rect(self.display, (0, 255, 0), (self.pipe_x+offset_x, self.pipe_gap_y + PIPE_GAP+offset_y, PIPE_WIDTH, WIN_HEIGHT - (self.pipe_gap_y + PIPE_GAP)))
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.display.blit(score_text, (10+offset_x, 10+offset_y))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Flappy Bird speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.") 