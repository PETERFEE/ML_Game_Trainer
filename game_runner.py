# game_runner.py
import pygame
import queue
import os
import torch
import numpy as np

# Import classes from other modules
from game_interface import BaseGameAI
from agent_core import Agent
import helper # For plotting functions
from utils import resource_path # <--- ADDED THIS IMPORT

# This device variable should match the one in agent_core.py
# (It's not strictly necessary to define it here if only used for model.to(device)
# which is handled by Agent, but it's fine to keep for clarity if needed elsewhere)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_game_thread(game_instance: BaseGameAI, agent_instance: Agent, 
                    game_command_queue: queue.Queue, plot_command_queue: queue.Queue, 
                    plot_scores: list, plot_records: list, plot_mean_scores: list):
    """
    Runs the main game loop and AI training in a separate thread.

    Args:
        game_instance (BaseGameAI): An instance of the selected game.
        agent_instance (Agent): The AI agent instance.
        game_command_queue (queue.Queue): Queue for sending commands to the UI.
        plot_command_queue (queue.Queue): Queue for sending plot updates to the UI.
        plot_scores (list): List to store scores of each game.
        plot_records (list): List to store the highest score achieved so far.
        plot_mean_scores (list): List to store the average score over all games.
    """
    total_score = 0
    record = 0
    running = True
    command = {}

    try:
        while running:
            # Process commands from Tkinter UI (non-blocking)
            try:
                command = game_command_queue.get_nowait()
                if command["command"] == "set_speed":
                    game_instance.set_speed(command["value"])
                elif command["command"] == "reset_game":
                    game_instance.reset()
                    agent_instance.n_games += 1
                    # When manually resetting, clear accumulated data
                    total_score = 0
                    plot_scores.clear()
                    plot_records.clear()
                    plot_mean_scores.clear()
                    record = 0 # Reset record on manual reset
                    print(f"Game manually reset. New Game No: {agent_instance.n_games}")
                elif command["command"] == "request_toggle_plot":
                    plot_command_queue.put({"command": "toggle_plot"})
                    print("Game thread: Requested plot toggle from main thread.")
                elif command["command"] == "quit_game":
                    print("Game thread: Received 'quit_game' command from UI. Terminating loop.")
                    running = False # Signal loop to terminate
            except queue.Empty:
                pass # No command, 'command' remains {}

            if not running: # Check if 'running' was set to False by a command
                break # Exit the while loop immediately

            # Process Pygame events (e.g., window close button)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Game thread: Pygame QUIT event detected. Signaling main thread for cleanup.")
                    plot_command_queue.put({"command": "quit_main"}) # Signal main thread to quit
                    running = False # Signal loop to terminate
                    break # Exit the for event loop

            if not running: # Check if 'running' was set to False by a Pygame event
                break # Exit the while loop immediately

            # --- Game Logic ---
            state_old = game_instance.get_state()
            action_index = agent_instance.get_action(state_old)
            reward, done, score = game_instance.play_step(action_index)
            state_new = game_instance.get_state()

            # Convert action_index to one-hot for agent's memory and training
            action_one_hot = [0] * agent_instance.output_size
            action_one_hot[action_index] = 1

            agent_instance.train_short_memory(state_old, action_one_hot, reward, state_new, done)
            agent_instance.remember(state_old, action_one_hot, reward, state_new, done)

            if done:
                game_instance.reset()
                # Only increment n_games if the game ended naturally (not manual reset)
                # The 'command' dict might not have a 'command' key if no UI command was received
                if command.get("command") != "reset_game":
                    agent_instance.n_games += 1
                agent_instance.train_long_memory()

                if score > record: # Update record only if current score is higher
                    record = score
                    # Save model specific to the game and model type
                    # FIX: Use resource_path from utils.py
                    model_save_path = resource_path(os.path.join('model', f"model_{game_instance.__class__.__name__}_{'cnn' if agent_instance.is_image_input else 'linear'}.pth"))
                    torch.save(agent_instance.model.state_dict(), model_save_path)
                    print(f"Model saved to: {model_save_path}")

                print('Game', agent_instance.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                plot_records.append(record) # Append the current record for plotting
                
                total_score += score
                mean_score = total_score / agent_instance.n_games if agent_instance.n_games > 0 else 0
                plot_mean_scores.append(mean_score) # Append mean score

                # Send plot update command to main thread
                plot_command_queue.put({
                    "command": "update_plot",
                    "scores": list(plot_scores), # Send copies of lists
                    "records": list(plot_records),
                    "mean_scores": list(plot_mean_scores)
                })

    finally: # Ensures this block runs when the try block is exited
        game_instance.close() # This calls pygame.quit() within the game thread
        print("Game thread terminated gracefully.")

