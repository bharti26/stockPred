import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from typing import Tuple, List, Dict, Any, Optional, Deque
from collections import deque
import random
from dataclasses import dataclass
import pandas as pd


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class AdvancedRL:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        memory_size: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
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
        self.data: Optional[pd.DataFrame] = None
        self.transaction_cost: float = 0.001
        self.initial_balance: float = 10000.0
        self.balance: float = self.initial_balance
        self.position: float = 0.0
        self.current_step: int = 0
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = [self.initial_balance]
        self.rewards: List[float] = []
        self.done: bool = False
        self.reward_scaling: float = 100.0

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.rewards = []
        self.done = False
        return np.zeros(self.state_size, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.done:
            raise RuntimeError("Episode has ended, call reset() to start a new episode")

        if self.data is None:
            raise ValueError("Data not initialized")

        current_price = self._get_current_price()
        reward = 0.0
        info: Dict[str, Any] = {}

        if action == 1:  # Buy
            if self.balance > 0:
                shares_to_buy = self.balance / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.position += shares_to_buy
                    reward = self._calculate_reward()
                    self.trades.append(
                        {
                            "type": "buy",
                            "price": current_price,
                            "shares": shares_to_buy,
                            "step": self.current_step,
                        }
                    )

        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                reward = self._calculate_reward()
                self.trades.append(
                    {
                        "type": "sell",
                        "price": current_price,
                        "shares": self.position,
                        "step": self.current_step,
                    }
                )
                self.position = 0

        # Update portfolio value
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        self.rewards.append(reward)

        # Prepare next state
        next_state = self._get_state()
        self.current_step += 1

        # Check if episode is done
        self.done = self._is_done()

        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "current_price": current_price,
        }

        return next_state, reward, self.done, info

    def _get_current_price(self) -> float:
        """Get the current price from the data."""
        if self.data is None:
            raise ValueError("Data not initialized")
        return float(self.data.iloc[self.current_step]["Close"])

    def _get_state(self) -> np.ndarray:
        """Get the current state representation."""
        if self.data is None:
            raise ValueError("Data not initialized")

        # Price data
        price_data = self.data.iloc[self.current_step]

        # Technical indicators
        rsi = self._calculate_rsi()
        macd = self._calculate_macd()
        bb_upper, bb_lower = self._calculate_bollinger_bands()

        # Position and balance info
        position_ratio = self.position * self._get_current_price() / self.initial_balance
        balance_ratio = self.balance / self.initial_balance

        # Combine all features
        state = np.array(
            [
                price_data["Close"] / price_data["Open"] - 1,  # Price change
                price_data["Volume"] / price_data["Volume"].mean() - 1,  # Volume change
                rsi / 100,  # Normalized RSI
                macd,  # MACD
                (price_data["Close"] - bb_lower) / (bb_upper - bb_lower),  # BB position
                position_ratio,  # Current position size
                balance_ratio,  # Current balance
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_reward(self) -> float:
        """Calculate the reward for the current step."""
        if len(self.portfolio_values) < 2:
            return 0.0
        current_value = self.balance + (self.position * self._get_current_price())
        prev_value = self.portfolio_values[-1]
        return ((current_value / prev_value) - 1) * self.reward_scaling

    def _is_done(self) -> bool:
        """Check if the episode is done."""
        if self.data is None:
            raise ValueError("Data not initialized")
        return (
            self.current_step >= len(self.data) - 1
            or (self.balance + (self.position * self._get_current_price()))
            < self.initial_balance * 0.5
        )

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if self.data is None:
            raise ValueError("Data not initialized")
        prices = self.data["Close"].iloc[max(0, self.current_step - period) : self.current_step + 1]
        deltas = np.diff(prices)
        gain = (deltas >= 0).astype(float) * deltas
        loss = (deltas < 0).astype(float) * (-deltas)

        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calculate_macd(self) -> float:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if self.data is None:
            raise ValueError("Data not initialized")
        prices = self.data["Close"].iloc[: self.current_step + 1]
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        return float(ema12.iloc[-1] - ema26.iloc[-1])

    def _calculate_bollinger_bands(self, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        if self.data is None:
            raise ValueError("Data not initialized")
        prices = self.data["Close"].iloc[max(0, self.current_step - period) : self.current_step + 1]
        sma = float(prices.mean())
        std = float(prices.std())
        return sma + (2 * std), sma - (2 * std)


class AdvancedDQNAgent:
    """Advanced DQN agent for stock trading."""

    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.memory: Deque[Experience] = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update = 1000
        self.steps = 0

        # Initialize networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def act(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax().item())

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Store experience in memory."""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def replay(self) -> None:
        """Train the network using prioritized experience replay."""
        if len(self.memory) < self.batch_size:
            return

        # Sample from memory
        batch = random.sample(list(self.memory), self.batch_size)
        states = torch.FloatTensor(np.array([exp.state for exp in batch]))
        actions = torch.LongTensor(np.array([exp.action for exp in batch]))
        rewards = torch.FloatTensor(np.array([exp.reward for exp in batch]))
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch]))
        dones = torch.FloatTensor(np.array([exp.done for exp in batch]))

        # Calculate current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

    def _build_network(self) -> nn.Module:
        """Build the neural network model."""
        return DuelingNetwork(self.state_size, self.action_size)

    def get_feature_importance(self, n_samples: int = 100) -> np.ndarray:
        """Calculate feature importance based on policy network weights.
        
        Args:
            n_samples: Number of samples to use for importance calculation
            
        Returns:
            Array of feature importance scores
        """
        # Get the weights from the first layer of the feature extraction network
        weights = self.policy_net.feature_layer[0].weight.detach().numpy()
        
        # Calculate importance as the absolute mean of weights for each input feature
        importance = np.abs(weights).mean(axis=0)
        
        # Normalize importance scores
        importance = importance / importance.sum()
        
        return importance


class DuelingNetwork(nn.Module):
    """Dueling network architecture for DQN."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.memory: List[Optional[Experience]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float,
    ) -> None:
        """Add experience to memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Sample a batch of experiences."""
        if len(self.memory) < batch_size:
            return None

        # Calculate sampling probabilities
        probs = self.priorities[: len(self.memory)] ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get experiences
        experiences = [self.memory[idx] for idx in indices if self.memory[idx] is not None]
        if len(experiences) != batch_size:
            return None

        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.memory)
