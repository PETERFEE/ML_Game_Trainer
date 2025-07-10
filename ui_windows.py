# ui_windows.py
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import sys
import matplotlib.pyplot as plt
from typing import Callable
import threading

import helper

# Global variable to store plot manager reference
_global_plot_manager = None

# Type for training callback function
StartTrainingCallback = Callable[[str], None]

class WelcomeWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk, start_training_callback: StartTrainingCallback):
        """
        Initializes the Welcome Window for game selection.
        """
        super().__init__(master)
        self.master = master
        self.start_training_callback = start_training_callback

        self.title("Welcome to AI Game Trainer")
        self.geometry("400x300")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.game_selection = tk.StringVar(value="snake")

        ttk.Label(self, text="Select a Game to Train:", font=("Arial", 16)).pack(pady=20)

        self.create_game_selection_buttons()
        ttk.Button(self, text="Start Training", command=self.start_training).pack(pady=20)

    def create_game_selection_buttons(self):
        """Creates the radio buttons for game selection."""
        radio_frame = ttk.Frame(self)
        radio_frame.pack(pady=10)

        ttk.Radiobutton(radio_frame, text="Snake Game", variable=self.game_selection, value="snake").pack(anchor="w")
        ttk.Radiobutton(radio_frame, text="Pong Game", variable=self.game_selection, value="pong").pack(anchor="w")

    def start_training(self):
        """Handles the 'Start Training' button click."""
        selected_game = self.game_selection.get()
        if not selected_game:
            messagebox.showwarning("No Game Selected", "Please select a game before starting training.")
            return

        print(f"User selected: {selected_game}. Starting training session...")
        self.destroy()

        self.master.after(100, lambda: self.start_training_callback(selected_game))

    def on_closing(self):
        """Handles the window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.master.destroy()
            sys.exit(0)


class InputControlWindow(tk.Toplevel):
    def __init__(self, master: tk.Toplevel, game_command_queue: queue.Queue, plot_command_queue: queue.Queue, shutdown_flag: threading.Event, training_callback: Callable[[str], None]):
        """
        Initializes the Input Control Window for game management.
        """
        super().__init__(master)
        self.title("Game AI Controls")
        self.geometry("350x250")
        self.game_queue = game_command_queue
        self.plot_queue = plot_command_queue
        self.shutdown_flag = shutdown_flag
        self.training_callback = training_callback
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root_window = master.master

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
        if self.shutdown_flag.is_set():
            return
            
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
        if self.shutdown_flag.is_set():
            return
            
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the game?")
        if confirm:
            try:
                import pygame
                self.game_queue.put({"command": "reset_game"})
                print("Control: Sent reset command for current game.")
            except:
                print("Control: Game window closed. Starting new training session...")
                self.destroy()
                try:
                    self.root_window.after(100, lambda: self.training_callback("snake"))
                except Exception as e:
                    print(f"Control: Error starting new training session: {e}")
                    self.root_window.destroy()
                    import sys
                    import subprocess
                    subprocess.Popen([sys.executable] + sys.argv)
                    sys.exit(0)

    def send_toggle_plot_command(self):
        """Sends a request to the game thread to toggle plot visibility."""
        if self.shutdown_flag.is_set():
            return
            
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
        """
        if self.shutdown_flag.is_set():
            return
            
        global _global_plot_manager

        while True:
            try:
                plot_cmd = self.plot_queue.get_nowait()
                if plot_cmd["command"] == "update_plot":
                    scores = plot_cmd["scores"]
                    records = plot_cmd["records"]
                    mean_scores = plot_cmd["mean_scores"]
                    _global_plot_manager = helper.plot(scores, records, mean_scores)
                elif plot_cmd["command"] == "toggle_plot":
                    if _global_plot_manager:
                        try:
                            helper.toggle_plot_visibility()
                        except Exception as e:
                            print(f"Main thread: Error toggling plot: {e}")
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
                    
                    self.shutdown_flag.set()
                    self.root_window.destroy()
                    return

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing plot command: {e}")
                if "destroyed" in str(e).lower():
                    print("Control: Application being destroyed, stopping queue checks.")
                    return
                break 
        
        if not self.shutdown_flag.is_set():
            self.after(100, self.check_plot_queue)

    def on_closing(self):
        """Handles the window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit the AI application?"):
            print("Control: Setting shutdown flag to terminate all threads...")
            self.shutdown_flag.set()
            
            print("Control: Sending quit command to game thread...")
            try:
                self.game_queue.put({"command": "quit_game"})
            except Exception as e:
                print(f"Control: Error sending quit command to game thread: {e}")

            print("Control: Sending quit command to main thread for UI/plot cleanup...")
            try:
                self.plot_queue.put({"command": "quit_main"})
            except Exception as e:
                print(f"Control: Error sending quit command to main thread: {e}")
            
            try:
                import pygame
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                print("Control: Posted Pygame QUIT event to close game window.")
            except Exception as e:
                print(f"Control: Could not post Pygame QUIT event: {e}")
            
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
                print("Control: Closed all matplotlib plots.")
            except Exception as e:
                print(f"Control: Could not close matplotlib plots: {e}")
            
            try:
                pass
            except:
                pass
            
            try:
                self.destroy()
            except Exception as e:
                print(f"Control: Error destroying control window: {e}")
            
            try:
                self.root_window.destroy()
            except Exception as e:
                print(f"Control: Error destroying root window: {e}")
            
            import sys
            sys.exit(0)

