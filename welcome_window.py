# welcome_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import sys

# This function will be provided by agent.py (passed in during init)
# It's a placeholder for the function that starts the training session.
# It's defined here for type hinting and clarity.
# from typing import Callable
# StartTrainingCallback = Callable[[str], None]

class WelcomeWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, start_training_callback):
        """
        Initializes the Welcome Window for game selection.

        Args:
            master (tk.Tk): The hidden root Tkinter window.
            start_training_callback (Callable[[str], None]): A function from agent.py
                                                              that will be called with the
                                                              selected game name to start training.
        """
        super().__init__(master)
        self.master = master
        self.start_training_callback = start_training_callback # Store the callback function

        self.title("Welcome to AI Game Trainer")
        self.geometry("400x300")
        # Set protocol for closing the window using the 'X' button
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.game_selection = tk.StringVar(value="snake") # Default selection

        ttk.Label(self, text="Select a Game to Train:", font=("Arial", 16)).pack(pady=20)

        self.create_game_selection_buttons()

        ttk.Button(self, text="Start Training", command=self.start_training).pack(pady=20)

    def create_game_selection_buttons(self):
        """Creates the radio buttons for game selection."""
        radio_frame = ttk.Frame(self)
        radio_frame.pack(pady=10)

        ttk.Radiobutton(radio_frame, text="Snake Game", variable=self.game_selection, value="snake").pack(anchor="w")
        ttk.Radiobutton(radio_frame, text="Pong Game", variable=self.game_selection, value="pong").pack(anchor="w")

        # Add more game buttons here as you implement them:
        # ttk.Radiobutton(radio_frame, text="My New Game", variable=self.game_selection, value="my_new_game_key").pack(anchor="w")

    def start_training(self):
        """Handles the 'Start Training' button click."""
        selected_game = self.game_selection.get()
        if not selected_game:
            messagebox.showwarning("No Game Selected", "Please select a game before starting training.")
            return

        print(f"User selected: {selected_game}. Starting training session...")
        self.destroy() # Close the welcome window

        # Call the callback function provided by agent.py to start training
        # Use master.after to schedule it on the main Tkinter thread
        self.master.after(100, lambda: self.start_training_callback(selected_game))

    def on_closing(self):
        """Handles the window close event (e.g., clicking 'X')."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.master.destroy() # Destroy the hidden root, which terminates the app
            sys.exit(0) # Ensure full exit in case mainloop is still running (for robustness)

