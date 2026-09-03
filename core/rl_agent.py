"""
core/rl_agent.py — Deep Q-Network (DQN) Agent for autonomous AMR navigation.

Implements an independent Q-learning agent with experience replay,
target network stabilization, and epsilon-greedy exploration.
"""

import os
import random
from collections import deque
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.rl_env import OBS_DIM, ACTIONS


class QNetwork(nn.Module):
    """Deep Q-Network for action value estimation."""

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Cyclic memory buffer for experience replay."""

    def __init__(self, capacity: int = 20000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Independent DQN Agent for decentralized obstacle avoidance & navigation."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.policy_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=25000)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select action via epsilon-greedy strategy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(self, batch_size: int = 64) -> float:
        """Perform one gradient descent step on a sampled experience batch."""
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str, reset_epsilon: bool = True, epsilon: Optional [float] = None) -> bool: # type: ignore
        """Load model weights from disk if available."""
        if not os.path.exists(filepath):
            return False
        try:
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if epsilon is not None:
                self.epsilon = epsilon
            elif reset_epsilon:
                self.epsilon = self.epsilon_min
            return True
        except Exception as e:
            print(f"[RLAgent] Failed to load {filepath}: {e}")
            return False

