# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from model import DQN, ReplayBuffer
import numpy as np

class DQNAgent:
    def __init__(self, stateDim, actionDim, lr=0.001, gamma=0.99, epsilon=1.0, epsilonDecay=0.995, epsilonMin=0.01):
        """Initialize the DQN agent with hyperparameters."""
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        #self.memory = deque(maxlen=10000)
        self.replayBuffer = ReplayBuffer(10000)
        self.model = DQN(stateDim, actionDim)
        self.targetModel = DQN(stateDim, actionDim)
        self.targetModel.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.targetModel.to(self.device)
        self.updateTargetEvery = 1000
        self.stepCounter = 0

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.actionDim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qValues = self.model(state)
        return qValues.argmax().item()

    """def remember(self, state, action, reward, nextState, done):
        Store experience in replay memory.
        self.memory.append((state, action, reward, nextState, done))"""

    def train(self, batchSize):
        """Train the model using a batch of experiences."""
        if len(self.replayBuffer) < batchSize:
            #print("Not enough samples to train")
            return
        batch = self.replayBuffer.sample(batchSize)
        # Unpack batch
        states, actions, rewards, nextStates, dones = zip(*batch)
        #print(f"Training")
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        nextStates = torch.FloatTensor(np.array(nextStates)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        qValues = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        nextQValues = self.targetModel(nextStates).max(1)[0]
        targetQ = rewards + (1 - dones) * self.gamma * nextQValues

        loss = nn.MSELoss()(qValues, targetQ)#.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.stepCounter += 1
        if self.stepCounter % self.updateTargetEvery == 0:
            self.updateTargetModel()
            #print(f"Updated target model at step {self.stepCounter}")
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        #self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

    def updateTargetModel(self):
        """Sync target model with the main model."""
        self.targetModel.load_state_dict(self.model.state_dict())

    def save(self, path):
        """Save the model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(path))
        self.targetModel.load_state_dict(self.model.state_dict())