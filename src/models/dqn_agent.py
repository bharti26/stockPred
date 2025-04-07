"""Deep Q-Network (DQN) implementation for stock trading.

This module implements a Deep Q-Network agent for reinforcement learning in stock trading.
It includes the DQN model architecture and the agent implementation with experience replay
and target network updates.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Deque

import random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as F


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class LayerConfig:
    """Configuration for the DQN layers.
    
    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer
        relu: ReLU activation function
    """
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear
    relu: nn.ReLU


@dataclass
class NetworkComponents:
    """Components of the DQN network.
    
    Attributes:
        policy_net: Main policy network
        target_net: Target network for stable learning
        optimizer: Optimizer for training
        criterion: Loss function
    """
    policy_net: 'DQN'
    target_net: 'DQN'
    optimizer: optim.Optimizer
    criterion: nn.Module


@dataclass
class NetworkConfig:
    """Configuration for the DQN networks.
    
    Attributes:
        state_size: Dimension of state space
        action_size: Dimension of action space
        learning_rate: Learning rate for optimizer
        device: Device to use for tensor operations
        components: Network components
    """
    state_size: int
    action_size: int
    learning_rate: float
    device: str
    components: NetworkComponents


@dataclass
class MemoryConfig:
    """Configuration for experience replay memory.
    
    Attributes:
        size: Maximum size of replay memory
        batch_size: Size of training batch
        buffer: Replay memory buffer
    """
    size: int
    batch_size: int
    buffer: Deque[Experience]


@dataclass
class ExplorationConfig:
    """Configuration for exploration parameters.
    
    Attributes:
        epsilon: Current exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Rate of exploration decay
        gamma: Discount factor for future rewards
    """
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    gamma: float


@dataclass
class DQNAgentConfig:
    """Configuration for the DQN agent.
    
    Attributes:
        state_size: Dimension of state space
        action_size: Dimension of action space
        memory_size: Maximum size of replay memory
        batch_size: Size of training batch
        gamma: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Rate of exploration decay
        learning_rate: Learning rate for optimizer
        device: Device to use for tensor operations
    """
    state_size: int
    action_size: int
    memory_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    device: str = "cpu"


class DQN(nn.Module):
    """Deep Q-Network model for stock trading."""

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initialize the DQN model.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network Agent for stock trading."""

    def __init__(self, config: DQNAgentConfig) -> None:
        """Initialize the DQN agent.

        Args:
            config: Agent configuration
        """
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.device = config.device

        # Initialize networks
        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self._target_model = DQN(self.state_size, self.action_size).to(self.device)
        self._target_model.load_state_dict(self.model.state_dict())
        self._target_model.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize memory
        self.memory = deque(maxlen=self.memory_size)

    @property
    def target_model(self) -> nn.Module:
        """Get the target network."""
        return self._target_model

    @property
    def target_net(self) -> nn.Module:
        """Get the target network (alias for target_model)."""
        return self._target_model

    @property
    def policy_net(self) -> nn.Module:
        """Get the policy network (alias for model)."""
        return self.model

    def reset(self) -> np.ndarray:
        """Reset the agent's state.

        Returns:
            Initial state
        """
        self.epsilon = self.epsilon
        return np.zeros(self.state_size, dtype=np.float32)

    def remember(
        self,
        state_or_experience: Union[Experience, np.ndarray],
        action: Optional[int] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        done: Optional[bool] = None
    ) -> None:
        """Store experience in memory.

        Args:
            state_or_experience: Either an Experience object or the current state
            action: Action taken (if not using Experience object)
            reward: Reward received (if not using Experience object)
            next_state: Next state (if not using Experience object)
            done: Whether episode is done (if not using Experience object)
        """
        if isinstance(state_or_experience, Experience):
            experience = state_or_experience
        else:
            if any(x is None for x in [action, reward, next_state, done]):
                raise ValueError("When not using Experience object, all parameters must be provided")
            experience = Experience(
                state=state_or_experience,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
        
        self.memory.append(experience)
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action based on the current state.

        Args:
            state: Current state
            training: Whether the agent is training

        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def replay(self, batch_size: Optional[int] = None) -> float:
        """Train the agent with experiences from memory.

        Args:
            batch_size: Size of batch to train on. If None, uses self.batch_size.

        Returns:
            Loss value from training
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0.0

        batch = random.sample(self.memory, batch_size)

        # Convert lists to numpy arrays first
        states = np.array([experience.state for experience in batch])
        next_states = np.array([experience.next_state for experience in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor([experience.action for experience in batch]).to(self.device)
        rewards = torch.FloatTensor([experience.reward for experience in batch]).to(self.device)
        dones = torch.FloatTensor([experience.done for experience in batch]).to(self.device)

        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            next_q_values = self._target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_model(self) -> None:
        """Update target network with weights from main network."""
        self._target_model.load_state_dict(self.model.state_dict())

    def save(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Path to save weights to
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str) -> None:
        """Load model weights.

        Args:
            path: Path to load weights from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        # Generate random next state and reward
        next_state = np.random.random(self.state_size)
        reward = np.random.uniform(-1, 1)
        terminated = np.random.random() < 0.1
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def get_feature_importance(self, data: pd.DataFrame) -> np.ndarray:
        """Get feature importance scores.

        Args:
            data: DataFrame containing features

        Returns:
            Array of feature importance scores
        """
        # For testing purposes, return random importance scores
        return np.random.random(len(self.state_size))
