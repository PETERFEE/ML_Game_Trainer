import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import os

from Linear_QNet import Linear_QNet
from cnn_qnet import CNN_QNet

from QTrainer import QTrainer
from helper import plot, toggle_plot_visibility
from utils import resource_path

from game_interface import BaseGameAI
from snake_game import SnakeGameAI
from pong_game import PongGameAI

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import sys
import matplotlib.pyplot as plt

import pygame

# NEW: Import the WelcomeWindow class
from welcome_window import WelcomeWindow

# Existing constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for PyTorch: {device}")

# Global queues for thread communication
game_command_queue = queue.Queue()
plot_command_queue = queue.Queue()

# Global variable to store plot manager reference (managed by the main thread)
_global_plot_manager = None


# --- Tkinter Input Control Window Class (no changes) ---
class InputControlWindow(tk.Toplevel):
    def __init__(self, master, game_queue_ref, plot_queue_ref):
        super().__init__(master)
        self.title("Game AI Controls")
        self.geometry("350x250")
        self.game_queue = game_queue_ref
        self.plot_queue = plot_queue_ref
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        ttk.Label(self, text="Game Speed (FPS):").pack(pady=5)
        self.speed_entry = ttk.Entry(self)
        self.speed_entry.insert(0, "100")
        self.speed_entry.pack(pady=5)
        ttk.Button(self, text="Set Speed", command=self.send_speed_command).pack(pady=5)

        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=10)

        self.plot_button_text = tk.StringVar(value="Hide Plot Window")
        self.plot_toggle_btn = ttk.Button(self, textvariable=self.plot_button_text, command=self.send_toggle_plot_command)
        self.plot_toggle_btn.pack(pady=5)
        self.is_plot_visible = True

        ttk.Button(self, text="Reset Game", command=self.send_reset_command).pack(pady=5)

        self.after(100, self.check_plot_queue)

    def send_speed_command(self):
        try:
            speed_val = int(self.speed_entry.get())
            if speed_val <= 0:
                messagebox.showwarning("Invalid Input", "Speed must be a positive integer.")
                return
            self.game_queue.put({"command": "set_speed", "value": speed_val})
            print(f"Control: Sent speed command: {speed_val}")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid integer for speed.")

    def send_reset_command(self):
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the game?")
        if confirm:
            self.game_queue.put({"command": "reset_game"})
            print("Control: Sent reset command.")

    def send_toggle_plot_command(self):
        self.game_queue.put({"command": "request_toggle_plot"})
        self.is_plot_visible = not self.is_plot_visible
        if self.is_plot_visible:
            self.plot_button_text.set("Hide Plot Window")
        else:
            self.plot_button_text.set("Show Plot Window")
        print("Control: Sent request to game thread to toggle plot.")

    def check_plot_queue(self):
        global _global_plot_manager

        while True:
            try:
                plot_cmd = self.plot_queue.get_nowait()
                if plot_cmd["command"] == "update_plot":
                    scores = plot_cmd["scores"]
                    records = plot_cmd["records"]
                    mean_scores = plot_cmd["mean_scores"]
                    _global_plot_manager = plot(scores, records, mean_scores)
                elif plot_cmd["command"] == "toggle_plot":
                    if _global_plot_manager:
                        toggle_plot_visibility()
                    else:
                        print("Main thread: Cannot toggle plot, manager not available yet.")
                elif plot_cmd["command"] == "quit_main":
                    print("Main thread: Received quit command. Initiating plot and UI shutdown.")
                    if _global_plot_manager:
                        try:
                            plt.close('all')
                            print("Main thread: Matplotlib figures closed.")
                        except Exception as e:
                            print(f"Error closing matplotlib figures: {e}")
                    
                    self.master.destroy() 
                    return

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing plot command: {e}")
                break 
        self.after(100, self.check_plot_queue)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the AI application?"):
            print("Control: Sending quit command to game thread...")
            self.game_queue.put({"command": "quit_game"})

            print("Control: Sending quit command to main thread for UI/plot cleanup...")
            self.plot_queue.put({"command": "quit_main"})
            


# --- Agent Class (no changes) ---
class Agent:
    def __init__(self, input_shape: tuple[int, ...], output_size: int, is_image_input: bool = False):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.input_shape = input_shape
        self.output_size = output_size
        self.is_image_input = is_image_input

        if self.is_image_input:
            if len(self.input_shape) != 3:
                raise ValueError("For image input, input_shape must be (channels, height, width).")
            self.model = CNN_QNet(self.input_shape, self.output_size).to(device)
            print(f"Agent using CNN_QNet with input shape {self.input_shape}")
        else:
            if len(self.input_shape) != 1:
                raise ValueError("For feature vector input, input_shape must be (num_features,).")
            self.model = Linear_QNet(self.input_shape[0], 256, self.output_size).to(device)
            print(f"Agent using Linear_QNet with input size {self.input_shape[0]}")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        model_file_path = resource_path(os.path.join('model', 'model.pth'))
        if os.path.exists(model_file_path):
            try:
                self.model.load_state_dict(torch.load(model_file_path, map_location=device))
                self.model.eval()
                print(f"Loaded trained model from: {model_file_path} to {device}")
            except Exception as e:
                print(f"Error loading model from {model_file_path}: {e}")
                print("Starting with a new untrained model.")
        else:
            print(f"No trained model found at {model_file_path}. Starting with a new untrained model.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> int:
        self.epsilon = 80 - self.n_games
        
        num_actions = self.output_size 
        
        final_move_idx = 0

        if random.randint(0, 200) < self.epsilon:
            final_move_idx = random.randint(0, num_actions - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)

            if not self.is_image_input:
                state0 = state0.unsqueeze(0)
            else:
                state0 = state0.unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                prediction = self.model(state0)
            final_move_idx = torch.argmax(prediction).item()

        self.model.train()
        return final_move_idx


# --- QTrainer Class (no changes) ---
class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        if isinstance(state, (tuple, list)):
            state_tensor = torch.tensor(np.array(state), dtype=torch.float).to(device)
            next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
            action_tensor = torch.tensor(np.array(action), dtype=torch.long).to(device)
            reward_tensor = torch.tensor(np.array(reward), dtype=torch.float).to(device)
            done_tensor = torch.tensor(np.array(done), dtype=torch.bool).to(device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device)
            action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(device)
            reward_tensor = torch.tensor(reward, dtype=torch.float).unsqueeze(0).to(device)
            done_tensor = torch.tensor(done, dtype=torch.bool).unsqueeze(0).to(device)

        pred = self.model(state_tensor)

        target = pred.clone()
        for idx in range(len(done_tensor)):
            Q_new = reward_tensor[idx]
            if not done_tensor[idx]:
                current_next_state_input = next_state_tensor[idx].unsqueeze(0)
                Q_new = reward_tensor[idx] + self.gamma * torch.max(self.model(current_next_state_input).detach())

            target[idx][torch.argmax(action_tensor[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# --- Function to run the Pygame Game Loop in a Thread ---
def run_game_thread(game_instance: BaseGameAI, agent_instance: Agent, game_queue_ref, plot_queue_ref, plot_scores, plot_records, plot_mean_scores):
    total_score = 0
    record = 0
    running = True
    command = {}

    try:
        while running:
            try:
                command = game_queue_ref.get_nowait()
                if command["command"] == "set_speed":
                    game_instance.set_speed(command["value"])
                elif command["command"] == "reset_game":
                    game_instance.reset()
                    agent_instance.n_games += 1
                    total_score = 0
                    plot_scores.clear()
                    plot_records.clear()
                    plot_mean_scores.clear()
                    record = 0
                    print(f"Game manually reset. New Game No: {agent_instance.n_games}")
                elif command["command"] == "request_toggle_plot":
                    plot_queue_ref.put({"command": "toggle_plot"})
                    print("Game thread: Requested plot toggle from main thread.")
                elif command["command"] == "quit_game":
                    print("Game thread: Received 'quit_game' command from UI.")
                    running = False
            except queue.Empty:
                pass

            if not running:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Game thread: Pygame QUIT event detected. Signaling main thread.")
                    plot_queue_ref.put({"command": "quit_main"})
                    running = False
                    break

            if not running:
                break

            state_old = game_instance.get_state()
            action_index = agent_instance.get_action(state_old)
            reward, done, score = game_instance.play_step(action_index)
            state_new = game_instance.get_state()

            action_one_hot = [0] * agent_instance.output_size
            action_one_hot[action_index] = 1

            agent_instance.train_short_memory(state_old, action_one_hot, reward, state_new, done)
            agent_instance.remember(state_old, action_one_hot, reward, state_new, done)

            if done:
                game_instance.reset()
                if command.get("command") != "reset_game":
                    agent_instance.n_games += 1
                agent_instance.train_long_memory()

                if score > record:
                    record = score
                    model_save_path = resource_path(os.path.join('model', f"model_{game_instance.__class__.__name__}_{'cnn' if agent_instance.is_image_input else 'linear'}.pth"))
                    torch.save(agent_instance.model.state_dict(), model_save_path)
                    print(f"Model saved to: {model_save_path}")

                print('Game', agent_instance.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                plot_records.append(record)
                
                total_score += score
                mean_score = total_score / agent_instance.n_games if agent_instance.n_games > 0 else 0
                plot_mean_scores.append(mean_score)

                plot_queue_ref.put({
                    "command": "update_plot",
                    "scores": list(plot_scores),
                    "records": list(plot_records),
                    "mean_scores": list(plot_mean_scores)
                })

    finally:
        game_instance.close()
        print("Game thread terminated gracefully.")


# --- Function to start the actual training session ---
def start_training_session(game_selection: str):
    """
    Initializes and runs the ML game training session based on user selection.
    This function is called after the WelcomeWindow handles game selection.
    """
    plot_scores = []
    plot_records = []
    plot_mean_scores = []

    game: BaseGameAI = None

    if game_selection == "snake":
        game = SnakeGameAI()
        print("Starting Snake Game AI (Feature-based)...")
    elif game_selection == "pong":
        game = PongGameAI()
        print("Starting Pong Game AI (Feature-based)...")
    else:
        raise ValueError(
            f"Unknown game selection: '{game_selection}'. "
            "Please choose an implemented game (e.g., 'snake' or 'pong')."
        )

    agent = Agent(
        input_shape=game.get_state_shape(),
        output_size=game.get_action_space_size(),
        is_image_input=game.get_state_is_image()
    )

    # The root Tkinter window is already created and hidden by the main 'train' function
    # We just need to create the control window as a Toplevel
    control_window_root = tk.Toplevel() # Create a new Toplevel window for controls
    control_window_root.withdraw() # Hide it initially if you prefer, or show it directly
    
    control_window = InputControlWindow(control_window_root, game_command_queue, plot_command_queue)
    control_window.deiconify() # Show the control window once created

    game_thread = threading.Thread(
        target=run_game_thread,
        args=(game, agent, game_command_queue, plot_command_queue, plot_scores, plot_records, plot_mean_scores)
    )
    game_thread.daemon = True
    game_thread.start()


# --- Main application entry point ---
def train():
    """
    Main function to start the application.
    Creates the hidden Tkinter root and displays the WelcomeWindow.
    """
    app_root = tk.Tk()
    app_root.withdraw() # Hide the main root window

    # Pass the start_training_session function as a callback to WelcomeWindow
    welcome_window = WelcomeWindow(app_root, start_training_session)
    welcome_window.deiconify() # Show the welcome window

    app_root.mainloop() 

    print("Tkinter main loop exited. Attempting to ensure all threads terminate...")
    
    # Check if game_thread was ever created and started before joining
    # This requires game_thread to be accessible in this scope.
    # A simple way is to make it a global variable or pass it around.
    # For now, we'll use a local check that might not always work if the thread
    # was never started or already terminated. A more robust solution for complex
    # apps might involve a dedicated "AppManager" class.
    
    # Check if the game_thread was created and is still alive.
    # This is a bit tricky with local scope. A more robust way would be to
    # store game_thread in a global variable or a dedicated application manager.
    # For now, this check might not always find it if it's already finished.
    # The daemon=True helps ensure the app exits even if it's hanging.
    
    # Re-evaluating the join logic here:
    # If the user quits from WelcomeWindow *before* starting training,
    # game_thread will not exist. The app_root.mainloop() will exit,
    # and the program should just terminate.
    # If training started, game_thread will be running.
    # The 'quit_main' command will destroy app_root, exiting mainloop.
    # Then we join the game_thread.
    
    # The simplest robust way is to make game_thread global or pass it.
    # Let's make it global for simplicity in this example.
    global game_thread_ref # Declare intention to use global variable
    if 'game_thread_ref' in globals() and game_thread_ref.is_alive():
        game_thread_ref.join(timeout=5) 
        if game_thread_ref.is_alive():
            print("Warning: Game thread did not terminate cleanly after timeout. It might be stuck.")
        else:
            print("Game thread confirmed terminated.")
    else:
        print("Game thread was not running or already terminated.")

    print("Application fully closed.")


if __name__ == '__main__':
    # Initialize game_thread_ref to None globally
    game_thread_ref = None 
    train()

