# main.py
import tkinter as tk
import threading
import queue
import sys

# Import modules containing our classes/functions
from ui_windows import WelcomeWindow, InputControlWindow # <--- NEW
from agent_core import Agent # <--- NEW
from game_runner import run_game_thread # <--- NEW

# Import game implementations
from game_interface import BaseGameAI
from snake_game import SnakeGameAI
from pong_game import PongGameAI


# Global queues for thread communication (defined here as central points)
game_command_queue = queue.Queue()
plot_command_queue = queue.Queue()

# Global reference for the game thread, to allow joining it on shutdown
game_thread_ref: threading.Thread = None


def start_training_session(game_selection: str):
    """
    Initializes and runs the ML game training session based on user selection.
    This function is called after the WelcomeWindow handles game selection.
    """
    global game_thread_ref # Declare intention to modify global variable

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

    # The root Tkinter window is already created and hidden by the main 'run_app' function
    # We just need to create the control window as a Toplevel
    control_window_root = tk.Toplevel() # Create a new Toplevel window for controls
    control_window_root.withdraw() # Hide it initially if you prefer, or show it directly
    
    control_window = InputControlWindow(control_window_root, game_command_queue, plot_command_queue)
    control_window.deiconify() # Show the control window once created

    # Create and start the game thread
    game_thread_ref = threading.Thread( # Assign to global variable
        target=run_game_thread,
        args=(game, agent, game_command_queue, plot_command_queue, plot_scores, plot_records, plot_mean_scores)
    )
    game_thread_ref.daemon = True # Daemon thread exits when main program exits
    game_thread_ref.start()


# --- Main application entry point ---
def run_app():
    """
    Main function to start the application.
    Creates the hidden Tkinter root and displays the WelcomeWindow.
    """
    app_root = tk.Tk()
    app_root.withdraw() # Hide the main root window

    # Pass the start_training_session function as a callback to WelcomeWindow
    welcome_window = WelcomeWindow(app_root, start_training_session)
    welcome_window.deiconify() # Show the welcome window

    # Start the Tkinter event loop on the main thread.
    # This blocks until app_root.destroy() is called (e.g., from WelcomeWindow.on_closing)
    app_root.mainloop() 

    # --- Code that executes AFTER the Tkinter main loop has exited ---
    print("Tkinter main loop exited. Attempting to ensure all threads terminate...")
    
    # Check if game_thread_ref was ever created and started before joining
    if game_thread_ref and game_thread_ref.is_alive(): # Check if it's not None and is alive
        game_thread_ref.join(timeout=5) 
        if game_thread_ref.is_alive():
            print("Warning: Game thread did not terminate cleanly after timeout. It might be stuck.")
        else:
            print("Game thread confirmed terminated.")
    else:
        print("Game thread was not running or already terminated.")

    print("Application fully closed.")


if __name__ == '__main__':
    run_app() # Call the new main entry point
