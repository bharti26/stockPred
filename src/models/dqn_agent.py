from collections import deque
from dataclasses import dataclass
import random
from typing import Optional, List, Union, Tuple, Deque, Dict

import numpy as np
import torch
from torch import nn, optim


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQN(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning Agent for stock trading."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,  # Force keyword arguments
        memory_size: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
        """Initialize DQN Agent.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            memory_size: Size of replay memory
            batch_size: Size of training batch
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for optimizer
            device: Device to use for tensor operations
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory: Deque[Experience] = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = device

        # Initialize networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _reshape_state(
        self, state: Union[np.ndarray, List[float], Tuple[float, ...], np.float32]
    ) -> np.ndarray:
        """Reshape state to match expected input shape.

        Args:
            state: State to reshape

        Returns:
            Reshaped state as a numpy array
        """
        # Handle tuple states from trading environment
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]  # Extract the state array from the tuple

        # Convert to numpy array
        if isinstance(state, (list, tuple)):
            state_array = np.array(state, dtype=np.float32)
        elif isinstance(state, (np.ndarray, np.float32)):
            state_array = np.array([state], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        # Reshape if needed
        if state_array.shape != (self.state_size,):
            state_array = state_array.flatten()[: self.state_size]
            if len(state_array) < self.state_size:
                pad_width = (0, self.state_size - len(state_array))
                state_array = np.pad(state_array, pad_width, "constant")

        # Ensure correct shape and type
        if not isinstance(state_array, np.ndarray):
            state_array = np.array(state_array, dtype=np.float32)
        if state_array.dtype != np.float32:
            state_array = state_array.astype(np.float32)
        if state_array.shape != (self.state_size,):
            state_array = state_array.reshape(self.state_size)

        return state_array

    def reset(self) -> np.ndarray:
        """Reset the agent's state.

        Returns:
            Initial state as a numpy array
        """
        self.epsilon = 1.0
        self.memory.clear()
        return np.zeros(self.state_size, dtype=np.float32)

    def act(self, state: Union[np.ndarray, List[float], Tuple[float, ...], np.float32]) -> int:
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(self.action_size))

        # Handle tuple state from trading environment
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]  # Extract the state array from the tuple

        state_array = self._reshape_state(state)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax().item())

    def remember(self, *args, **kwargs) -> None:
        """Store experience in replay memory.

        Args:
            *args: Either a single Experience object or individual components
            **kwargs: Keyword arguments for individual components
        """
        if len(args) == 1 and isinstance(args[0], Experience):
            self.memory.append(args[0])
        elif len(args) == 5:
            state, action, reward, next_state, done = args
            # Handle tuple states from trading environment
            if isinstance(state, tuple) and len(state) == 2:
                state = state[0]
            if isinstance(next_state, tuple) and len(next_state) == 2:
                next_state = next_state[0]
            state_array = self._reshape_state(state)
            next_state_array = self._reshape_state(next_state)
            self.memory.append(Experience(state_array, action, reward, next_state_array, done))
        elif all(k in kwargs for k in ["state", "action", "reward", "next_state", "done"]):
            state = kwargs["state"]
            next_state = kwargs["next_state"]
            # Handle tuple states from trading environment
            if isinstance(state, tuple) and len(state) == 2:
                state = state[0]
            if isinstance(next_state, tuple) and len(next_state) == 2:
                next_state = next_state[0]
            state_array = self._reshape_state(state)
            next_state_array = self._reshape_state(next_state)
            self.memory.append(
                Experience(
                    state_array,
                    kwargs["action"],
                    kwargs["reward"],
                    next_state_array,
                    kwargs["done"],
                )
            )
        else:
            raise ValueError(
                "Invalid arguments. Provide either an Experience object or all individual components"
            )

    def replay(self, batch_size: Optional[int] = None) -> float:
        """Train the model using experience replay.

        Args:
            batch_size: Size of batch for training.
                If None, uses self.batch_size.

        Returns:
            float: The loss value from training.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0.0

        # Sample a random batch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Convert states and next_states to numpy arrays before converting to tensors
        states = np.array([exp.state for exp in minibatch], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in minibatch], dtype=np.float32)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor([exp.action for exp in minibatch]).to(self.device)
        rewards_tensor = torch.FloatTensor([exp.reward for exp in minibatch]).to(self.device)
        dones_tensor = torch.FloatTensor([exp.done for exp in minibatch]).to(self.device)

        # Get Q-values for current states and next states
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        next_q_values = self.target_net(next_states_tensor).max(1)[0].detach()

        # Compute target Q-values
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        # Compute loss and update policy network
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def update_target_model(self) -> None:
        """Update target network weights with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save model weights
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to load model weights from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple containing:
                - next_state: Next state
                - reward: Reward received
                - terminated: Whether episode is terminated
                - truncated: Whether episode is truncated
                - info: Additional information
        """
        # This is a placeholder for the actual environment step
        # In a real implementation, this would interact with the environment
        next_state = np.random.random(self.state_size)
        reward = float(np.random.randn())
        terminated = bool(np.random.random() < 0.1)
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    @property
    def target_model(self) -> DQN:
        """Get the target network.

        Returns:
            The target DQN model
        """
        return self.target_net
