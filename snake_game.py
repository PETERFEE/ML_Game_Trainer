# snake_game.py

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys # Keep sys import for potential use, but not for exiting in play_step

from utils import resource_path
from game_interface import BaseGameAI # <--- NEW: Import the abstract base class

# Other constants remain
BLOCK_SIZE = 20
FONT_COLOR = (255, 255, 255) # White
FOOD_COLOR = (200, 0, 0)     # Red
SNAKE_COLOR1 = (0, 255, 0)   # Green
SNAKE_COLOR2 = (0, 200, 0)   # Darker Green
BACKGROUND_COLOR = (0, 0, 0) # Black

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Changed to inherit from BaseGameAI
class SnakeGameAI(BaseGameAI):

    def __init__(self, w=640, h=480):
        # Initialize all pygame modules
        pygame.init()

        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Make speed an instance variable
        self.current_speed = 100 # Default speed, can be changed via input window

        # Set game icon
        icon_file = resource_path('ai.ico')
        game_icon = pygame.image.load(icon_file)
        pygame.display.set_icon(game_icon)

        # Load font as instance variable
        font_file = resource_path('arial.ttf')
        self.font = pygame.font.Font(font_file, 25)

        self.reset()

    # Implements BaseGameAI.reset()
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # No return value needed for reset, just sets internal state

    # Implements BaseGameAI.get_state()
    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right (relative to current direction)
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left (relative to current direction)
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
            ]
        return np.array(state, dtype=int)

    # Implements BaseGameAI.play_step(action_index)
    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        
        # Map integer action_index to game-specific action array [straight, right, left]
        # 0: straight, 1: right, 2: left
        action_array = [0, 0, 0]
        action_array[action_index] = 1

        self._move(action_array) # update the head
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        
        # Check for collision or timeout
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 # Penalty for game over

        # Check if food was eaten (only if game is not over from collision/timeout)
        if not game_over and self.head == self.food:
            self.score += 1
            reward = 10 # Reward for eating food
            self._place_food()
        elif not game_over: # Only pop if game not over and no food eaten
            self.snake.pop() # Remove tail if no food eaten

        self.render() # Call the render method (which calls _update_ui)
        self.clock.tick(self.current_speed) # Use the instance variable for speed

        return reward, game_over, self.score

    # is_collision method: Already good, just formally implements the interface.
    def is_collision(self, pt=None) -> bool:
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    # Implements BaseGameAI.is_game_over()
    def is_game_over(self) -> bool:
        # This method is used by the agent to check if the game has ended
        return self.is_collision() or self.frame_iteration > 100*len(self.snake)

    # Implements BaseGameAI.get_action_space_size()
    def get_action_space_size(self) -> int:
        return 3 # For [straight, right, left]

    # Implements BaseGameAI.get_state_shape()
    def get_state_shape(self) -> tuple[int, ...]:
        # Based on your get_state() method which returns an 11-element array
        return (11,)

    # Implements BaseGameAI.get_state_is_image()
    def get_state_is_image(self) -> bool:
        return False # This game uses a feature vector, not an image

    # Implements BaseGameAI.get_score()
    def get_score(self) -> int:
        return self.score

    # Implements BaseGameAI.render()
    def render(self) -> None:
        self._update_ui() # Calls your existing UI update method

    # Implements BaseGameAI.close()
    def close(self) -> None:
        pygame.quit() # Cleanly uninitialize Pygame

    # _place_food method: (No changes needed)
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # _update_ui method: (No changes needed)
    def _update_ui(self):
        self.display.fill(BACKGROUND_COLOR)

        for i, pt in enumerate(self.snake):
            color = SNAKE_COLOR1 if i == 0 else SNAKE_COLOR2 # Head is SNAKE_COLOR1
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, FONT_COLOR)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # _move method: (No changes needed)
    def _move(self, action: list[int]): # Expects one-hot array from play_step
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)

    # Implements BaseGameAI.set_speed()
    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0: # Ensure speed is positive
                self.current_speed = new_speed
                print(f"Snake speed updated to: {self.current_speed}")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.")

