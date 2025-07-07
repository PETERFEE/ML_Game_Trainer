# QTrainer.py
import torch
import torch.nn as nn
import numpy as np # <--- ADDED THIS IMPORT

class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() 

    def train_step(self, state, action, reward, next_state, done):
        # Convert all to tensors and move to device
        # Handle single sample vs. batch for conversion
        # The tensors are expected to already be on the correct device when passed here
        if isinstance(state, (tuple, list)): # If it's a batch from zip(*mini_sample)
            state_tensor = torch.tensor(np.array(list(state)), dtype=torch.float)
            next_state_tensor = torch.tensor(np.array(list(next_state)), dtype=torch.float)
            action_tensor = torch.tensor(np.array(list(action)), dtype=torch.long)
            reward_tensor = torch.tensor(np.array(list(reward)), dtype=torch.float)
            done_tensor = torch.tensor(np.array(list(done)), dtype=torch.bool)
        else: # Single sample
            state_tensor = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
            next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float).unsqueeze(0)
            action_tensor = torch.tensor(np.array(action), dtype=torch.long).unsqueeze(0)
            reward_tensor = torch.tensor(np.array(reward), dtype=torch.float).unsqueeze(0)
            done_tensor = torch.tensor(np.array(done), dtype=torch.bool).unsqueeze(0)

        # Ensure tensors are on the same device as the model
        # Assuming model has a .device attribute or is already on device
        # The Agent class should ensure the state is on the correct device before calling train_short/long_memory
        state_tensor = state_tensor.to(self.model.device) 
        next_state_tensor = next_state_tensor.to(self.model.device)
        action_tensor = action_tensor.to(self.model.device)
        reward_tensor = reward_tensor.to(self.model.device)
        done_tensor = done_tensor.to(self.model.device)

        # 1: predicted Q values with current state
        pred = self.model(state_tensor)

        target = pred.clone()
        for idx in range(len(done_tensor)):
            Q_new = reward_tensor[idx]
            if not done_tensor[idx]:
                Q_new = reward_tensor[idx] + self.gamma * torch.max(self.model(next_state_tensor[idx].unsqueeze(0)).detach()) 
            
            # Update the Q_new at the action taken
            target[idx][torch.argmax(action_tensor[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

