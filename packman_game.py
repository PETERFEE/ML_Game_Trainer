import pygame
import numpy as np
import random
from game_interface import BaseGameAI

# Packman constants
WIN_WIDTH = 320
WIN_HEIGHT = 320
GRID_SIZE = 16
ROWS = WIN_HEIGHT // GRID_SIZE
COLS = WIN_WIDTH // GRID_SIZE
PLAYER_COLOR = (255, 255, 0)
GHOST_COLORS = [(255, 0, 0), (255, 192, 203), (0, 0, 255)]  # Red, Pink, Blue
BEAN_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)

class PackmanGameAI(BaseGameAI):
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption('Packman - Close via UI window')
        self.clock = pygame.time.Clock()
        self.current_speed = 10
        self.font = pygame.font.SysFont('Arial', 18)
        self.reset()

    def reset(self):
        self.player_pos = [ROWS // 2, COLS // 2]
        self.ghost_positions = [[0, 0], [ROWS-1, 0], [0, COLS-1]]  # 3 ghosts at corners
        self.score = 0
        self.frame_iteration = 0
        self.is_game_over_flag = False
        self.beans = set()
        
        # Create all possible bean positions (excluding player and ghost positions)
        all_positions = []
        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) != tuple(self.player_pos) and (i, j) not in [tuple(pos) for pos in self.ghost_positions]:
                    all_positions.append((i, j))
        
        # Randomly select 1/3 of the beans
        num_beans = len(all_positions) // 3
        selected_beans = random.sample(all_positions, num_beans)
        self.beans = set(selected_beans)

    def get_state(self) -> np.ndarray:
        # State: player x/y, 3 ghosts x/y, nearest bean x/y, distance to nearest ghost, distance to bean
        px, py = self.player_pos
        gx1, gy1 = self.ghost_positions[0]
        gx2, gy2 = self.ghost_positions[1]
        gx3, gy3 = self.ghost_positions[2]
        if self.beans:
            nearest_bean = min(self.beans, key=lambda b: abs(b[0]-px)+abs(b[1]-py))
            bx, by = nearest_bean
            dist_bean = abs(bx-px) + abs(by-py)
        else:
            bx, by = -1, -1
            dist_bean = 0
        # Distance to nearest ghost
        dist_ghost1 = abs(gx1-px) + abs(gy1-py)
        dist_ghost2 = abs(gx2-px) + abs(gy2-py)
        dist_ghost3 = abs(gx3-px) + abs(gy3-py)
        dist_ghost = min(dist_ghost1, dist_ghost2, dist_ghost3)
        state = [
            px / ROWS,
            py / COLS,
            gx1 / ROWS,
            gy1 / COLS,
            gx2 / ROWS,
            gy2 / COLS,
            gx3 / ROWS,
            gy3 / COLS,
            bx / ROWS,
            by / COLS,
            dist_ghost / (ROWS+COLS),
            dist_bean / (ROWS+COLS)
        ]
        return np.array(state, dtype=np.float32)

    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        self.frame_iteration += 1
        reward = -0.01  # small penalty for each step
        done = False
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.display = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        # Actions: 0=up, 1=down, 2=left, 3=right
        dx, dy = 0, 0
        if action_index == 0:
            dx, dy = -1, 0
        elif action_index == 1:
            dx, dy = 1, 0
        elif action_index == 2:
            dx, dy = 0, -1
        elif action_index == 3:
            dx, dy = 0, 1
        # Move player
        new_px = max(0, min(ROWS-1, self.player_pos[0]+dx))
        new_py = max(0, min(COLS-1, self.player_pos[1]+dy))
        self.player_pos = [new_px, new_py]
        # Eat bean
        if tuple(self.player_pos) in self.beans:
            self.beans.remove(tuple(self.player_pos))
            self.score += 1
            reward = 1.0
        # Move all 3 ghosts (random walk towards player)
        px, py = self.player_pos
        for i in range(3):
            gx, gy = self.ghost_positions[i]
            if i == 0:  # Red ghost: 70% chase, 30% random
                if random.random() < 0.7:
                    if gx < px: gx += 1
                    elif gx > px: gx -= 1
                    elif gy < py: gy += 1
                    elif gy > py: gy -= 1
                else:
                    if random.random() < 0.5:
                        gx = max(0, min(ROWS-1, gx + random.choice([-1, 1])))
                    else:
                        gy = max(0, min(COLS-1, gy + random.choice([-1, 1])))
            elif i == 1:  # Pink ghost: chase only if player within 1/3 radius
                distance = abs(gx - px) + abs(gy - py)
                radius_threshold = (ROWS + COLS) // 3
                if distance <= radius_threshold:
                    if gx < px: gx += 1
                    elif gx > px: gx -= 1
                    elif gy < py: gy += 1
                    elif gy > py: gy -= 1
                else:
                    if random.random() < 0.5:
                        gx = max(0, min(ROWS-1, gx + random.choice([-1, 1])))
                    else:
                        gy = max(0, min(COLS-1, gy + random.choice([-1, 1])))
            else:  # Blue ghost: 100% random movement
                if random.random() < 0.5:
                    gx = max(0, min(ROWS-1, gx + random.choice([-1, 1])))
                else:
                    gy = max(0, min(COLS-1, gy + random.choice([-1, 1])))
            self.ghost_positions[i] = [gx, gy]
        # Check collision with any ghost
        if self.player_pos in self.ghost_positions:
            done = True
            reward = -5.0
            self.is_game_over_flag = True
        # Win if all beans eaten
        if not self.beans:
            done = True
            reward = 10.0
            self.is_game_over_flag = True
        self.render()
        self.clock.tick(self.current_speed)
        return reward, done, self.score

    def is_game_over(self) -> bool:
        return self.is_game_over_flag

    def get_action_space_size(self) -> int:
        return 4  # up, down, left, right

    def get_state_shape(self) -> tuple[int, ...]:
        return (12,)  # player x/y, 3 ghosts x/y, bean x/y, distances

    def get_state_is_image(self) -> bool:
        return False

    def get_score(self) -> int:
        return self.score

    def render(self) -> None:
        win_w, win_h = self.display.get_size()
        offset_x = (win_w - WIN_WIDTH) // 2 if win_w > WIN_WIDTH else 0
        offset_y = (win_h - WIN_HEIGHT) // 2 if win_h > WIN_HEIGHT else 0
        self.display.fill(BG_COLOR)
        # Draw beans
        for (i, j) in self.beans:
            pygame.draw.circle(self.display, BEAN_COLOR, (j*GRID_SIZE+GRID_SIZE//2+offset_x, i*GRID_SIZE+GRID_SIZE//2+offset_y), 3)
        # Draw player
        pygame.draw.circle(self.display, PLAYER_COLOR, (self.player_pos[1]*GRID_SIZE+GRID_SIZE//2+offset_x, self.player_pos[0]*GRID_SIZE+GRID_SIZE//2+offset_y), GRID_SIZE//2)
        # Draw all 3 ghosts with different colors (square shape)
        for i, ghost_pos in enumerate(self.ghost_positions):
            pygame.draw.rect(self.display, GHOST_COLORS[i], (ghost_pos[1]*GRID_SIZE+offset_x, ghost_pos[0]*GRID_SIZE+offset_y, GRID_SIZE, GRID_SIZE))
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, (10+offset_x, 10+offset_y))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def set_speed(self, new_speed: int) -> None:
        try:
            new_speed = int(new_speed)
            if new_speed > 0:
                self.current_speed = new_speed
                print(f"Packman speed updated to: {self.current_speed} FPS")
            else:
                print(f"Ignored invalid speed: {new_speed}. Must be positive.")
        except ValueError:
            print(f"Invalid speed value received: {new_speed}. Must be an integer.") 