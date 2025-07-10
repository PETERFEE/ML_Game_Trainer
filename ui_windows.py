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
        Initializes the Welcome Window with introduction and game selection.
        """
        super().__init__(master)
        self.master = master
        self.start_training_callback = start_training_callback

        self.title("ML Game Trainer - AI Learning Platform")
        self.geometry("600x500")
        self.resizable(True, True)  # Allow window to expand to full screen
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.game_selection = tk.StringVar(value="snake")
        self.current_page = tk.StringVar(value="intro")

        self.create_welcome_interface()

    def create_welcome_interface(self):
        """Creates the main welcome interface with tabs."""
        # Main title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill='x', pady=(20, 10))
        ttk.Label(title_frame, text="🎮 ML Game Trainer", font=("Arial", 20, "bold")).pack()
        ttk.Label(title_frame, text="Advanced AI Learning Platform", font=("Arial", 12)).pack()

        # Tab control
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Introduction tab
        intro_frame = ttk.Frame(notebook)
        notebook.add(intro_frame, text="📖 Introduction")
        self.create_introduction_page(intro_frame)

        # Game selection tab
        game_frame = ttk.Frame(notebook)
        notebook.add(game_frame, text="🎯 Game Selection")
        self.create_game_selection_page(game_frame)

        # Usage guide tab
        guide_frame = ttk.Frame(notebook)
        notebook.add(guide_frame, text="📋 Usage Guide")
        self.create_usage_guide_page(guide_frame)

        # Contact tab
        contact_frame = ttk.Frame(notebook)
        notebook.add(contact_frame, text="📬 Contact")
        self.create_contact_page(contact_frame)

        # Remove the global Start Training button (was here previously)
        # button_frame = ttk.Frame(self)
        # button_frame.pack(fill='x', pady=20)
        # ttk.Button(button_frame, text="🚀 Start Training", command=self.start_training).pack(pady=10)

    def create_introduction_page(self, parent):
        """Creates the introduction page with project overview."""
        # Scrollable text widget
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Introduction content
        intro_text = """
🤖 What is ML Game Trainer?

ML Game Trainer is a cutting-edge artificial intelligence platform that demonstrates 
machine learning in action through classic video games. Our system uses Deep Q-Learning 
(DQN) algorithms to train AI agents to master various games autonomously.

🎯 Key Features:
• 7 Classic Games: Snake, Pong, Flappy Bird, Tetris, Galaga, Dino Run, Packman
• Real-time AI Training: Watch neural networks learn in real-time
• Advanced Algorithms: Deep Q-Learning with CNN and Linear networks
• Interactive Controls: Adjust speed, reset games, toggle visualizations
• Model Persistence: Automatic saving and loading of trained models

🧠 How It Works:

1. NEURAL NETWORKS: Our AI uses both Convolutional Neural Networks (CNN) for 
   image-based games and Linear networks for vector-based games.

2. REINFORCEMENT LEARNING: The AI learns through trial and error, receiving 
   rewards for good actions and penalties for mistakes.

3. DEEP Q-LEARNING: Combines deep learning with Q-learning to handle complex 
   state spaces and discover optimal strategies.

4. REAL-TIME VISUALIZATION: Watch training progress through live plots showing 
   scores, records, and learning curves.

🔬 Scientific Importance:

This platform demonstrates fundamental concepts in:
• Artificial Intelligence and Machine Learning
• Reinforcement Learning algorithms
• Neural Network architectures
• Game Theory and Strategy optimization
• Real-time data visualization

🚀 Future Applications:

The techniques demonstrated here have applications in:
• Autonomous robotics and navigation
• Game AI development
• Financial trading algorithms
• Medical diagnosis systems
• Autonomous vehicle control
• Industrial process optimization

This project serves as both an educational tool and a foundation for more 
advanced AI research and development.
        """
        
        text_widget = tk.Text(scrollable_frame, wrap=tk.WORD, width=70, height=20, 
                             font=("Arial", 10), bg="white", relief="flat")
        text_widget.pack(padx=20, pady=20, fill='both', expand=True)
        text_widget.insert("1.0", intro_text)
        text_widget.config(state=tk.DISABLED)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_game_selection_page(self, parent):
        """Creates the game selection page."""
        ttk.Label(parent, text="Select a Game to Train:", font=("Arial", 16, "bold")).pack(pady=20)
        
        # Game descriptions
        game_info = {
            "snake": "🐍 Snake Game - Classic pathfinding with food collection",
            "pong": "🏓 Pong Game - Two-player AI with improved rewards",
            "flappy": "🐦 Flappy Bird - Continuous flight mechanics",
            "tetris": "🧩 Tetris - Block stacking and line clearing",
            "galaga": "🚀 Galaga - Space shooter with enemy waves",
            "dino": "🦖 Dino Run - Endless runner with obstacles",
            "packman": "👻 Packman - Maze navigation with 3 smart ghosts"
        }

        radio_frame = ttk.Frame(parent)
        radio_frame.pack(pady=20, padx=40, fill='x')

        for game, description in game_info.items():
            frame = ttk.Frame(radio_frame)
            frame.pack(fill='x', pady=5)
            ttk.Radiobutton(frame, text=description, variable=self.game_selection, 
                           value=game).pack(anchor="w")

        # Add Start Training button inside radio_frame, below all games
        start_btn = ttk.Button(radio_frame, text="🚀 Start Training", command=self.start_training)
        start_btn.pack(pady=(20, 0))

    def create_usage_guide_page(self, parent):
        """Creates the usage guide page."""
        guide_text = """
📋 How to Use ML Game Trainer:

🎮 Getting Started:
1. Select a game from the Game Selection tab
2. Click "Start Training" to begin AI learning
3. Watch the AI train in real-time through game and plot windows

🎛️ Controls During Training:
• Game Speed: Adjust FPS to control training speed
• Reset Game: Start fresh training session
• Hide/Show Plot: Toggle training visualization
• Close Windows: Proper shutdown sequence

📊 Understanding the Interface:
• Game Window: Shows AI playing the selected game
• Plot Window: Displays training metrics (scores, records, averages)
• Control Window: Adjust settings and manage training

🧠 Training Process:
• The AI starts with random actions
• Gradually learns optimal strategies through trial and error
• Training progress is saved automatically
• Models improve over time with more training

💡 Tips for Best Results:
• Let the AI train for at least 100+ games for good performance
• Higher speeds (200+ FPS) for faster training
• Monitor the plot to see learning progress
• Different games may require different training times

🔧 Technical Details:
• Uses PyTorch for neural networks
• Deep Q-Learning algorithm
• Automatic model saving/loading
• Real-time visualization with Matplotlib
• Multi-threaded architecture for smooth performance

🎯 Expected Outcomes:
• Snake: Learns efficient pathfinding
• Pong: Develops paddle positioning skills
• Flappy Bird: Masters timing and navigation
• Tetris: Optimizes piece placement
• Galaga: Improves shooting accuracy
• Dino Run: Enhances obstacle avoidance
• Packman: Develops strategic navigation
        """
        
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        text_widget = tk.Text(scrollable_frame, wrap=tk.WORD, width=70, height=20, 
                             font=("Arial", 10), bg="white", relief="flat")
        text_widget.pack(padx=20, pady=20, fill='both', expand=True)
        text_widget.insert("1.0", guide_text)
        text_widget.config(state=tk.DISABLED)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_contact_page(self, parent):
        """Creates the contact page with author info."""
        contact_text = """
Contact Information

Name: Peter Feng
Phone: 4698262067
Email: fenghon000@gmail.com
GitHub: PETERFEE
"""
        label = ttk.Label(parent, text=contact_text, font=("Arial", 12), justify="left")
        label.pack(padx=30, pady=40, anchor="nw")

    def create_game_selection_buttons(self):
        """Legacy method - now handled by create_game_selection_page."""
        pass

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
        self.resizable(True, True)  # Allow window to expand to full screen
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
                    # Only destroy Tk windows if shutdown was initiated by on_closing
                    if hasattr(self, '_pending_final_destroy') and self._pending_final_destroy:
                        def delayed_destroy():
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
                        self.after(500, delayed_destroy)  # 500ms delay
                        self.shutdown_flag.set()  # Set shutdown flag after scheduling destroy
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
            self._pending_final_destroy = True  # Add a flag to track pending destruction

            print("Control: Sending quit command to game thread...")
            try:
                self.game_queue.put({"command": "quit_game"})
            except Exception as e:
                print(f"Control: Error sending quit command to game thread: {e}")

            try:
                import pygame
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                print("Control: Posted Pygame QUIT event to close game window.")
            except Exception as e:
                print(f"Control: Could not post Pygame QUIT event: {e}")

            # Wait 300ms before closing the UI window
            def close_ui_window():
                print("Control: Closing UI window after game window.")
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

            # Wait 600ms before sending quit_main to plot window (to ensure UI is closed first)
            def send_plot_quit():
                print("Control: Sending quit command to main thread for UI/plot cleanup...")
                try:
                    self.plot_queue.put({"command": "quit_main"})
                except Exception as e:
                    print(f"Control: Error sending quit command to main thread: {e}")
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    print("Control: Closed all matplotlib plots.")
                except Exception as e:
                    print(f"Control: Could not close matplotlib plots: {e}")

            self.after(300, close_ui_window)
            self.after(600, send_plot_quit)

            # Do NOT destroy Tk windows or plot immediately; wait for the scheduled functions

