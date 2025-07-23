import torch
import torch.nn as nn
from collections import deque
import random

"""
class DQN(nn.Module):
    def __init__(self, inputDim, outputDim):
        Initialise the DQN with input and output dimensions.
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputDim)
        )

    def forward(self, x):
        Forward pass through the network.
        return self.net(x)
"""    

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Initialise the DQN with input and output dimensions."""
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)