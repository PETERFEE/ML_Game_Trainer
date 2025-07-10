# main.py
import tkinter as tk
import threading
import queue
import sys
from typing import Optional

from ui_windows import WelcomeWindow, InputControlWindow
from agent_core import Agent
from game_runner import run_game_thread

from game_interface import BaseGameAI
from snake_game import SnakeGameAI
from pong_game import PongGameAI

# Global queues for thread communication
game_command_queue = queue.Queue()
plot_command_queue = queue.Queue()

# Global reference for the game thread
game_thread_ref: Optional[threading.Thread] = None

# Global shutdown flag to coordinate between threads
shutdown_flag = threading.Event()


def start_training_session(game_selection: str):
    """
    Initializes and runs the ML game training session based on user selection.
    """
    global game_thread_ref, shutdown_flag
    
    # Reset shutdown flag for new session
    shutdown_flag.clear()

    plot_scores = []
    plot_records = []
    plot_mean_scores = []

    game: Optional[BaseGameAI] = None

    if game_selection == "snake":
        game = SnakeGameAI()
        print("Starting Snake Game AI...")
    elif game_selection == "pong":
        game = PongGameAI()
        print("Starting Pong Game AI...")
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

    # Create control window
    control_window_root = tk.Toplevel()
    control_window_root.withdraw()
    
    control_window = InputControlWindow(control_window_root, game_command_queue, plot_command_queue, shutdown_flag, start_training_session)
    control_window.deiconify()

    # Create and start the game thread
    game_thread_ref = threading.Thread(
        target=run_game_thread,
        args=(game, agent, game_command_queue, plot_command_queue, plot_scores, plot_records, plot_mean_scores, shutdown_flag)
    )
    game_thread_ref.daemon = True
    game_thread_ref.start()


def run_app():
    """
    Main function to start the application.
    """
    app_root = tk.Tk()
    app_root.withdraw()

    welcome_window = WelcomeWindow(app_root, start_training_session)
    welcome_window.deiconify()

    app_root.mainloop() 

    # Cleanup after Tkinter main loop exits
    print("Tkinter main loop exited. Cleaning up threads...")
    
    # Set shutdown flag to signal all threads to stop
    shutdown_flag.set()
    
    # Check if game_thread_ref was ever created and started before joining
    if game_thread_ref and game_thread_ref.is_alive():
        game_thread_ref.join(timeout=5) 
        if game_thread_ref.is_alive():
            print("Warning: Game thread did not terminate cleanly after timeout.")
        else:
            print("Game thread confirmed terminated.")
    else:
        print("Game thread was not running or already terminated.")

    print("Application fully closed.")


if __name__ == '__main__':
    run_app()
