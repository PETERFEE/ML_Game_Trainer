import pygame
import numpy as np
import random
from game_interface import BaseGameAI

# Tetris constants
WIN_WIDTH = 200
WIN_HEIGHT = 400
BLOCK_SIZE = 20
BOARD_WIDTH = WIN_WIDTH // BLOCK_SIZE
BOARD_HEIGHT = WIN_HEIGHT // BLOCK_SIZE

# Tetromino shapes (I, O, T, S, Z, J, L)
TETROMINOS = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
]

class TetrisGameAI(BaseGameAI):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Tetris - Close via UI window')
        self.clock = pygame.time.Clock()
        self.current_speed = 10
        self.font = pygame.font.SysFont('Arial', 18)
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False
        self.spawn_piece()

    def spawn_piece(self):
        self.piece_type = random.randint(0, len(TETROMINOS) - 1)
        self.piece = np.array(TETROMINOS[self.piece_type])
        self.piece_x = BOARD_WIDTH // 2 - self.piece.shape[1] // 2
        self.piece_y = 0
        self.piece_rotation = 0

    def get_state(self) -> np.ndarray:
        # State: flattened board + piece type (one-hot) + piece x position (normalized) + piece y position (normalized)
        board_flat = self.board.flatten() / 1.0
        piece_onehot = np.zeros(len(TETROMINOS))
        piece_onehot[self.piece_type] = 1
        state = np.concatenate([
            board_flat,
            piece_onehot,
            [self.piece_x / BOARD_WIDTH, self.piece_y / BOARD_HEIGHT]
        ])
        return state.astype(np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = 0.01  # small reward for staying alive
        done = False

        # Actions: 0 = left, 1 = right, 2 = rotate, 3 = drop
        if action_index == 0:
            self.try_move(-1, 0)
        elif action_index == 1:
            self.try_move(1, 0)
        elif action_index == 2:
            self.try_rotate()
        elif action_index == 3:
            while self.try_move(0, 1):
                pass
            self.lock_piece()

        # Gravity
        if not self.try_move(0, 1):
            self.lock_piece()

        # Check for game over
        if np.any(self.board[0] != 0):
            done = True
            reward = -5.0
            self.is_game_over_flag = True

        self.render()
        self.clock.tick(self.current_speed)
        return reward, done, self.score

    def try_move(self, dx, dy):
        new_x = self.piece_x + dx
        new_y = self.piece_y + dy
        if self.valid_position(self.piece, new_x, new_y):
            self.piece_x = new_x
            self.piece_y = new_y
            return True
        return False

    def try_rotate(self):
        new_piece = np.rot90(self.piece)
        if self.valid_position(new_piece, self.piece_x, self.piece_y):
            self.piece = new_piece

    def valid_position(self, piece, x, y):
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i, j]:
                    if (x + j < 0 or x + j >= BOARD_WIDTH or y + i >= BOARD_HEIGHT or
                        (y + i >= 0 and self.board[y + i, x + j])):
                        return False
        return True

    def lock_piece(self):
        for i in range(self.piece.shape[0]):
            for j in range(self.piece.shape[1]):
                if self.piece[i, j]:
                    if self.piece_y + i < 0:
                        self.is_game_over_flag = True
                        return
                    self.board[self.piece_y + i, self.piece_x + j] = 1
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        lines_cleared = 0
        new_board = []
        for row in self.board:
            if np.all(row):
                lines_cleared += 1
            else:
                new_board.append(row)
        while len(new_board) < BOARD_HEIGHT:
            new_board.insert(0, np.zeros(BOARD_WIDTH))
        self.board = np.array(new_board)
        self.score += lines_cleared

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 4  # left, right, rotate, drop

    def get_state_shape(self) -> tuple[int, ...]:
        # board + piece one-hot + x + y
        return (BOARD_WIDTH * BOARD_HEIGHT + len(TETROMINOS) + 2,)

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        self.display.fill((0, 0, 0))
        # Draw board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y, x]:
                    pygame.draw.rect(self.display, (0, 255, 255), (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Draw current piece
        for i in range(self.piece.shape[0]):
            for j in range(self.piece.shape[1]):
                if self.piece[i, j]:
                    pygame.draw.rect(self.display, (255, 255, 0), ((self.piece_x + j) * BLOCK_SIZE, (self.piece_y + i) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
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
                print(f"Tetris speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.") 