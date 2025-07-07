# ui_windows.py
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import sys
import matplotlib.pyplot as plt # Needed for plt.close('all')
from typing import Callable # For type hinting callbacks

# Import helper for plotting functions
import helper

# Global variable to store plot manager reference (managed by the main thread)
# This is accessed by InputControlWindow and set by helper.plot
_global_plot_manager = None

# A placeholder for the function that starts the training session,
# which will be passed from main.py
StartTrainingCallback = Callable[[str], None]

class WelcomeWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, start_training_callback: StartTrainingCallback):
        """
        Initializes the Welcome Window for game selection.

        Args:
            master (tk.Tk): The hidden root Tkinter window.
            start_training_callback (StartTrainingCallback): A function from main.py
                                                              that will be called with the
                                                              selected game name to start training.
        """
        super().__init__(master)
        self.master = master
        self.start_training_callback = start_training_callback # Store the callback function

        self.title("Welcome to AI Game Trainer")
        self.geometry("400x300")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.game_selection = tk.StringVar(value="snake") # Default selection

        ttk.Label(self, text="Select a Game to Train:", font=("Arial", 16)).pack(pady=20)

        # Game selection radio buttons
        self.create_game_selection_buttons()

        ttk.Button(self, text="Start Training", command=self.start_training).pack(pady=20)

    def create_game_selection_buttons(self):
        """Creates the radio buttons for game selection."""
        radio_frame = ttk.Frame(self)
        radio_frame.pack(pady=10)

        ttk.Radiobutton(radio_frame, text="Snake Game", variable=self.game_selection, value="snake").pack(anchor="w")
        ttk.Radiobutton(radio_frame, text="Pong Game", variable=self.game_selection, value="pong").pack(anchor="w")

        # Add more game buttons here as you implement them
        # ttk.Radiobutton(radio_frame, text="New Game", variable=self.game_selection, value="new_game_key").pack(anchor="w")

    def start_training(self):
        """Handles the 'Start Training' button click."""
        selected_game = self.game_selection.get()
        if not selected_game:
            messagebox.showwarning("No Game Selected", "Please select a game before starting training.")
            return

        print(f"User selected: {selected_game}. Starting training session...")
        self.destroy() # Close the welcome window

        # Call the callback function provided by main.py to start training
        # Use master.after to schedule it on the main Tkinter thread
        self.master.after(100, lambda: self.start_training_callback(selected_game))

    def on_closing(self):
        """Handles the window close event (e.g., clicking 'X')."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.master.destroy() # Destroy the hidden root, which terminates the app
            sys.exit(0) # Ensure full exit in case mainloop is still running


class InputControlWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, game_command_queue: queue.Queue, plot_command_queue: queue.Queue):
        """
        Initializes the Input Control Window for game management.

        Args:
            master (tk.Tk): The hidden root Tkinter window.
            game_command_queue (queue.Queue): Queue for sending commands to the game thread.
            plot_command_queue (queue.Queue): Queue for receiving plot commands from the game thread.
        """
        super().__init__(master)
        self.title("Game AI Controls")
        self.geometry("350x250")
        self.game_queue = game_command_queue
        self.plot_queue = plot_command_queue
        self.protocol("WM_DELETE_WINDOW", self.on_closing) # Override default close behavior

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
        """Sends a command to set the game speed."""
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
        """Sends a command to reset the game."""
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the game?")
        if confirm:
            self.game_queue.put({"command": "reset_game"})
            print("Control: Sent reset command.")

    def send_toggle_plot_command(self):
        """Sends a request to the game thread to toggle plot visibility."""
        self.game_queue.put({"command": "request_toggle_plot"})
        self.is_plot_visible = not self.is_plot_visible
        if self.is_plot_visible:
            self.plot_button_text.set("Hide Plot Window")
        else:
            self.plot_button_text.set("Show Plot Window")
        print("Control: Sent request to game thread to toggle plot.")

    def check_plot_queue(self):
        """
        Checks the plot command queue for updates.
        Runs on the main Tkinter thread.
        """
        # Access the global plot manager defined in main.py or ui_windows.py
        # For simplicity, we'll assume it's accessible via helper.plot's internal mechanism
        # or that the global _global_plot_manager is imported from main.py or set here.
        # Let's ensure it's accessible via helper directly.
        global _global_plot_manager # This will be set by helper.plot()

        while True:
            try:
                plot_cmd = self.plot_queue.get_nowait()
                if plot_cmd["command"] == "update_plot":
                    scores = plot_cmd["scores"]
                    records = plot_cmd["records"]
                    mean_scores = plot_cmd["mean_scores"]
                    # Call helper.plot which will set _global_plot_manager internally
                    _global_plot_manager = helper.plot(scores, records, mean_scores)
                elif plot_cmd["command"] == "toggle_plot":
                    if _global_plot_manager: # Check if plot manager exists before toggling
                        helper.toggle_plot_visibility()
                    else:
                        print("Main thread: Cannot toggle plot, manager not available yet.")
                elif plot_cmd["command"] == "quit_main":
                    print("Main thread: Received quit command. Initiating plot and UI shutdown.")
                    if _global_plot_manager:
                        try:
                            plt.close('all') # Close all matplotlib figures
                            print("Main thread: Matplotlib figures closed.")
                        except Exception as e:
                            print(f"Error closing matplotlib figures: {e}")
                    
                    self.master.destroy() # Destroy the Tkinter root window
                    return # Exit this loop as master is being destroyed

            except queue.Empty:
                break # No more commands, stop checking for now
            except Exception as e:
                print(f"Error processing plot command: {e}")
                break 
        self.after(100, self.check_plot_queue) # Schedule next check

    def on_closing(self):
        """Handles the window close event (e.g., clicking 'X')."""
        if messagebox.askokcancel("Quit", "Do you want to quit the AI application?"):
            print("Control: Sending quit command to game thread...")
            self.game_queue.put({"command": "quit_game"}) # Signal the game thread to stop

            print("Control: Sending quit command to main thread for UI/plot cleanup...")
            # Send a command back to the main thread's queue
            # This ensures the Tkinter window and plot are closed on their owning thread.
            self.plot_queue.put({"command": "quit_main"})
            # DO NOT CALL self.master.destroy() or sys.exit(0) directly here.
            # The 'quit_main' command handled by check_plot_queue will perform master.destroy().
            # This ensures a graceful shutdown for all GUI components.

