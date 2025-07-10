# game_runner.py
import pygame
import queue
import os
import torch
import numpy as np
import threading

from game_interface import BaseGameAI
from agent_core import Agent
import helper
from utils import resource_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_game_thread(game_instance: BaseGameAI, agent_instance: Agent, 
                    game_command_queue: queue.Queue, plot_command_queue: queue.Queue, 
                    plot_scores: list, plot_records: list, plot_mean_scores: list, shutdown_flag: threading.Event):
    """
    Runs the main game loop and AI training in a separate thread.
    """
    total_score = 0
    record = 0
    running = True
    command = {}

    try:
        while running and not shutdown_flag.is_set():
            try:
                command = game_command_queue.get_nowait()
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
                    plot_command_queue.put({"command": "toggle_plot"})
                    print("Game thread: Requested plot toggle from main thread.")
                elif command["command"] == "quit_game":
                    print("Game thread: Received 'quit_game' command from UI. Terminating loop.")
                    running = False
            except queue.Empty:
                pass

            if not running or shutdown_flag.is_set():
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Game thread: Pygame QUIT event detected. Signaling main thread for cleanup.")
                    plot_command_queue.put({"command": "quit_main"})
                    running = False
                    break

            if not running or shutdown_flag.is_set():
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

                if not shutdown_flag.is_set():
                    plot_command_queue.put({
                        "command": "update_plot",
                        "scores": list(plot_scores),
                        "records": list(plot_records),
                        "mean_scores": list(plot_mean_scores)
                    })

    finally:
        game_instance.close()
        print("Game thread terminated gracefully.")

