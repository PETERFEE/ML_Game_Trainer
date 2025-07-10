# agent_core.py
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import os

from Linear_QNet import Linear_QNet
from cnn_qnet import CNN_QNet
from QTrainer import QTrainer
from utils import resource_path

# Constants (can be moved to a config.py if more constants are added)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Detect device for PyTorch (CPU or CUDA GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for PyTorch: {device} (from agent_core.py)")


class Agent:
    def __init__(self, input_shape: tuple[int, ...], output_size: int, is_image_input: bool = False):
        self.n_games = 0
        self.epsilon = 1.0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=100_000) # popleft()
        self.batch_size = 128 # Larger batch size
        self.lr = 0.0005  # Lower learning rate for Pong
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Slower decay for more exploration

        self.input_shape = input_shape
        self.output_size = output_size
        self.is_image_input = is_image_input

        # Initialize the correct model based on input type
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

        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

        # Model loading logic (consider making model saving/loading more robust for different games)
        try:
            model_path = resource_path(os.path.join('model', 'model.pth'))
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded trained model from: {model_path} to {device}")
        except RuntimeError as e:
            if 'size mismatch' in str(e) or 'shapes cannot be multiplied' in str(e):
                print(f"Model shape mismatch for {model_path}: {e}\nDeleting incompatible model file and starting with a new untrained model.")
                try:
                    os.remove(model_path)
                    print(f"Deleted incompatible model file: {model_path}")
                except Exception as del_e:
                    print(f"Failed to delete model file: {model_path}. Error: {del_e}")
            else:
                print(f"Error loading model from {model_path}: {e}\nStarting with a new untrained model.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}\nStarting with a new untrained model.")

    def remember(self, state, action, reward, next_state, done):
        """Stores a transition in the agent's memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Trains the model using a batch from long-term memory."""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the model using a single recent transition."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> int:
        """
        Chooses an action based on the current state using an epsilon-greedy policy.
        Returns an integer representing the chosen action index.
        """
        # Trade-off exploration / exploitation
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        num_actions = self.output_size 
        final_move_idx = 0 # Default to action 0

        if random.randint(0, 200) < self.epsilon:
            # Explore: choose a random action
            final_move_idx = random.randint(0, num_actions - 1)
        else:
            # Exploit: choose the action with the highest predicted Q-value
            state0 = torch.tensor(state, dtype=torch.float).to(device)

            if not self.is_image_input:
                # For linear models: add a batch dimension (1, num_features)
                state0 = state0.unsqueeze(0)
            else:
                # For CNN: state should already be (channels, height, width) from game.get_state().
                # Add a batch dimension: (1, channels, height, width)
                state0 = state0.unsqueeze(0)

            self.model.eval() # Set model to evaluation mode for inference
            with torch.no_grad(): # Disable gradient calculation for inference
                prediction = self.model(state0)
            final_move_idx = torch.argmax(prediction).item()
        
        self.model.train() # Set model back to training mode
        return final_move_idx

