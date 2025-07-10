# snake_game.py

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys
from typing import Optional

from utils import resource_path
from game_interface import BaseGameAI

BLOCK_SIZE = 20
FONT_COLOR = (255, 255, 255)
FOOD_COLOR = (200, 0, 0)
SNAKE_COLOR1 = (0, 255, 0)
SNAKE_COLOR2 = (0, 200, 0)
BACKGROUND_COLOR = (0, 0, 0)

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI(BaseGameAI):

    def __init__(self, w=640, h=480):
        pygame.init()

        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        pygame.display.set_caption('Snake Game - Close via UI window')
        
        self.clock = pygame.time.Clock()
        self.current_speed = 100

        icon_file = resource_path('ai.ico')
        game_icon = pygame.image.load(icon_file)
        pygame.display.set_icon(game_icon)

        font_file = resource_path('arial.ttf')
        self.font = pygame.font.Font(font_file, 25)

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration = 0

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

        if self.food is None:
            self._place_food()
        
        assert self.food is not None, "Food should be placed by _place_food()"

        state = [
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y
            ]
        return np.array(state, dtype=int)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        
        action_array = [0, 0, 0]
        action_array[action_index] = 1

        self._move(action_array)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10

        if not game_over and self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        elif not game_over:
            self.snake.pop()

        self.render()
        self.clock.tick(self.current_speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None) -> bool:
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def is_game_over(self) -> bool:
        return self.is_collision() or self.frame_iteration > 100*len(self.snake)

    def get_action_space_size(self) -> int:
        return 3

    def get_state_shape(self) -> tuple[int, ...]:
        return (11,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        win_w, win_h = self.display.get_size()
        offset_x = (win_w - self.w) // 2 if win_w > self.w else 0
        offset_y = (win_h - self.h) // 2 if win_h > self.h else 0
        self.display.fill(BACKGROUND_COLOR)

        for i, pt in enumerate(self.snake):
            color = SNAKE_COLOR1 if i == 0 else SNAKE_COLOR2
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x+offset_x, pt.y+offset_y, BLOCK_SIZE, BLOCK_SIZE))

        if self.food is not None:
            pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x+offset_x, self.food.y+offset_y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, FONT_COLOR)
        self.display.blit(text, [0+offset_x, 0+offset_y])
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        self.display.fill(BACKGROUND_COLOR)

        for i, pt in enumerate(self.snake):
            color = SNAKE_COLOR1 if i == 0 else SNAKE_COLOR2
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        if self.food is not None:
            pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, FONT_COLOR)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: list[int]):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
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

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Snake speed updated to: {self.current_speed}")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.")

