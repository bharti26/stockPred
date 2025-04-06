from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym/Gymnasium.

    This environment simulates stock trading, where an agent can buy, sell,
    or hold a single stock. The state space includes the stock price data,
    technical indicators, and the agent's account information.

    The environment follows the gym interface with custom:
    - Observation space: Account info + market features
    - Action space: Discrete(3) for buy (1), sell (2), hold (0)
    - Reward function: Change in portfolio value

    Attributes:
        df (pd.DataFrame): Historical price data and indicators
        initial_balance (float): Starting account balance
        balance (float): Current account balance
        shares_held (int): Number of shares currently held
        current_step (int): Current step in the episode
        trades (list): List of executed trades
        action_space (spaces.Discrete): Action space
        observation_space (spaces.Box): Observation space
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        """Initialize the trading environment.

        Args:
            df (pd.DataFrame): Historical price data with technical indicators
            initial_balance (float, optional): Starting account balance. Defaults to 10000.0
        """
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.trades: List[Dict[str, Any]] = []
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # buy (1), sell (2), hold (0)

        # Observation space: [balance, shares_held, current_price, position_value,
        # technical_indicators]
        n_features = len([col for col in df.columns if 'norm' in col])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + n_features,),
            dtype=np.float32
        )

        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        This method is called at the beginning of each episode. It resets the
        account balance, shares held, and current step to their initial values.

        Args:
            seed (int, optional): Random seed. Defaults to None.
            options (dict, optional): Additional options. Defaults to None.

        Returns:
            tuple: Initial observation and empty info dict
        """
        super().reset(seed=seed, options=options)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.trades = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        This method processes the agent's action (buy/sell/hold), updates the
        environment state, and calculates the reward based on the change in
        portfolio value.

        Args:
            action (int): Action to take (0: hold, 1: buy, 2: sell)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                observation (np.ndarray): Current state observation
                reward (float): Reward from the action
                terminated (bool): Whether the episode is done
                truncated (bool): Whether the episode was truncated
                info (dict): Additional information
        """
        self.current_step += 1
        current_price = float(self.df.iloc[self.current_step]['Close'])

        # Calculate reward
        reward = 0.0
        done = False

        if action == 1:  # Buy
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': current_price
                })

        elif action == 2:  # Sell
            if self.shares_held > 0:
                value = self.shares_held * current_price
                self.balance += value
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.shares_held,
                    'price': current_price
                })
                self.shares_held = 0

        # Calculate portfolio value and reward
        portfolio_value = self.balance + (self.shares_held * current_price)
        if len(self.trades) > 0:
            reward = portfolio_value - self.initial_balance

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self) -> np.ndarray:
        """Get the current state observation.

        The observation includes:
        - Current balance
        - Number of shares held
        - Current stock price
        - Current position value
        - Normalized technical indicators

        Returns:
            np.ndarray: Current state observation
        """
        current_price = float(self.df.iloc[self.current_step]['Close'])

        # Get normalized technical indicators
        tech_cols = [col for col in self.df.columns if 'norm' in col]
        tech_indicators = self.df.iloc[self.current_step][tech_cols].values.astype(np.float32)

        # Combine all observations and create a numpy array with explicit type
        obs = np.array([
            float(self.balance),
            float(self.shares_held),
            float(current_price),
            float(self.shares_held * current_price),  # Current position value
        ], dtype=np.float32)

        # Concatenate with technical indicators
        obs = np.concatenate([obs, tech_indicators])

        return obs

    def render(self, mode: str = 'human') -> None:
        """Render the environment to the screen.

        Displays current step, balance, shares held, current price,
        and total portfolio value.

        Args:
            mode (str, optional): Rendering mode. Defaults to 'human'.
        """
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance:.2f}')
            print(f'Shares held: {self.shares_held}')
            current_price = float(self.df.iloc[self.current_step]["Close"])
            print(f'Current price: {current_price:.2f}')
            portfolio_value = self.balance + (self.shares_held * current_price)
            print(f'Portfolio value: {portfolio_value:.2f}')
