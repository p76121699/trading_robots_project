import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import os
class Actor(nn.Module):
    def __init__(self, state_dim, num_stocks, num_actions):
        super(Actor, self).__init__()
        hidden_dim = 64
        hidden_dim2 = 128
        self.num_stocks = num_stocks
        self.num_actions = num_actions
        
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim2, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks * num_actions)
        )
        
    def forward(self, state):
        batch_size = state.size(0)
        rnn_output, _ = self.lstm(state)  # [B, T, D]
        hidden_state = rnn_output[:, -1, :]
        out = self.net(hidden_state)
        out = out.view(batch_size, self.num_stocks, self.num_actions)
        return out  

class ReinforceAgent:
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_gap = 1000
        self.num_stocks = getattr(args, 'num_stocks', 20)
        self.action_dim = getattr(args, 'num_actions', 11)
        self.lr = getattr(args, 'actor_lr', 1e-6)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.state_dim, self.num_stocks, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.trajectory = []  # (log_prob, reward)

    def decode_action(self, action_idx):
        actions = []
        for idx in action_idx:
            if idx < self.action_dim // 2:
                action_type = 2  # Sell
                quantity = (self.action_dim // 2 - idx) * self.action_gap
            elif idx > self.action_dim // 2:
                action_type = 1  # Buy
                quantity = (idx - self.action_dim // 2) * self.action_gap
            else:
                action_type = 0  # Hold
                quantity = 0
            actions.append((action_type, quantity))
        return actions
    
    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0)  # [1, state_dim]

        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        action = self.decode_action(action_idx.squeeze(0).cpu().numpy())
        return action, log_prob

    def store_transition(self, log_prob, reward):
        self.trajectory.append((log_prob, reward))

    def update(self):
        R = 0
        returns = []
        for _, reward in reversed(self.trajectory):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

        policy_loss = []
        for (log_prob, _), G in zip(self.trajectory, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.trajectory.clear()
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)
        
    def load_model(self, path):
        if os.path.isfile(path):
            self.actor.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
            print("Model loaded from:", path)
        else:
            print("No model file found at:", path)
