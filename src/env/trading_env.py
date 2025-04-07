"""Stock trading environment for reinforcement learning.

This module implements a Gymnasium-compatible environment for stock trading.
It provides a simulation environment where an agent can learn to trade stocks
by taking actions (buy, sell, hold) and receiving rewards based on portfolio performance.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class TradingState:
    """Represents the current state of a trading environment.
    
    Attributes:
        balance (float): Current account balance
        shares_held (int): Number of shares currently held
        trades (list): List of executed trades
        current_step (int): Current step in the episode
    """
    balance: float
    shares_held: int
    trades: List[Dict[str, Any]]
    current_step: int


class StockTradingEnv(gym.Env):
    """Stock trading environment for reinforcement learning."""

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0) -> None:
        """Initialize the environment.

        Args:
            df: DataFrame containing stock data
            initial_balance: Initial account balance
        """
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.state = TradingState(initial_balance, 0, [], 0)
        self.trades = []

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # Define feature names for the observation space
        self.feature_names = [
            'balance',
            'shares_held',
            'current_price',
            'current_step',
            'num_trades',
            'initial_balance',
            'RSI_norm',
            'MACD_norm',
            'Signal_norm',
            'BB_Upper_norm',
            'BB_Middle_norm',
            'BB_Lower_norm',
            'Volume_norm',
            'Close_norm',
            'High_norm',
            'Low_norm',
            'Open_norm'
        ]

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            Tuple containing initial observation and info dictionary
        """
        self.state = TradingState(self.initial_balance, 0, [], 0)
        self.trades = []
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: Action to take (0: hold, 1: buy, 2: sell)

        Returns:
            Tuple containing:
            - observation: Current state observation
            - reward: Reward for the action taken
            - terminated: Whether the episode is terminated
            - truncated: Whether the episode was truncated
            - info: Additional information
        """
        current_price = self.df.iloc[self.state.current_step]["Close"]
        reward = 0
        terminated = False
        truncated = False

        if action == 1:  # Buy
            shares_to_buy = int(self.state.balance / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.state.balance -= cost
                self.state.shares_held += shares_to_buy
                self.trades.append({
                    "step": self.state.current_step,
                    "action": "buy",
                    "shares": shares_to_buy,
                    "price": current_price,
                })

        elif action == 2:  # Sell
            if self.state.shares_held > 0:
                value = self.state.shares_held * current_price
                self.state.balance += value
                self.trades.append({
                    "step": self.state.current_step,
                    "action": "sell",
                    "shares": self.state.shares_held,
                    "price": current_price,
                })
                self.state.shares_held = 0

        # Calculate reward
        portfolio_value = self.state.balance + (self.state.shares_held * current_price)
        reward = portfolio_value - self.initial_balance

        # Update step
        self.state.current_step += 1

        # Check if episode is done
        if self.state.current_step >= len(self.df) - 1:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self) -> np.ndarray:
        """Get the current observation.

        Returns:
            Current state observation
        """
        current_price = self.df.iloc[self.state.current_step]["Close"]
        return np.array([
            self.state.balance,
            self.state.shares_held,
            current_price,
            self.state.current_step,
            len(self.trades),
            self.initial_balance,
        ], dtype=np.float32)

    def render(self, mode: str = 'human') -> None:
        """Render the environment to the screen.

        Displays current step, balance, shares held, current price,
        and total portfolio value.

        Args:
            mode (str, optional): Rendering mode. Defaults to 'human'.
        """
        if mode == 'human':
            print(f'Step: {self.state.current_step}')
            print(f'Balance: {self.state.balance:.2f}')
            print(f'Shares held: {self.state.shares_held}')
            current_price = float(self.df.iloc[self.state.current_step]["Close"])
            print(f'Current price: {current_price:.2f}')
            portfolio_value = self.state.balance + (self.state.shares_held * current_price)
            print(f'Portfolio value: {portfolio_value:.2f}')
