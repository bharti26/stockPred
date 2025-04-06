import numpy as np
import pandas as pd
import pytest

from src.env.trading_env import StockTradingEnv


@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, len(dates)),
            "High": np.linspace(105, 115, len(dates)),
            "Low": np.linspace(95, 105, len(dates)),
            "Close": np.linspace(102, 112, len(dates)),
            "Volume": np.random.randint(1000, 2000, len(dates)),
            "Close_norm": np.random.random(len(dates)),
            "Volume_norm": np.random.random(len(dates)),
        },
        index=dates,
    )


@pytest.fixture
def trading_env(sample_data):
    """Create trading environment instance"""
    initial_balance = 10000.0
    return StockTradingEnv(sample_data, initial_balance)


def test_initialization(trading_env):
    """Test if environment is correctly initialized"""
    assert trading_env.initial_balance == 10000.0
    assert trading_env.balance == 10000.0
    assert trading_env.shares_held == 0
    assert trading_env.current_step == 0
    assert len(trading_env.trades) == 0


def test_reset(trading_env):
    """Test environment reset"""
    # Make some changes to the environment
    trading_env.balance = 5000.0
    trading_env.shares_held = 10
    trading_env.current_step = 5
    # Reset the environment
    observation, info = trading_env.reset()
    # Check if state is properly reset
    assert trading_env.balance == trading_env.initial_balance
    assert trading_env.shares_held == 0
    assert trading_env.current_step == 0
    assert len(trading_env.trades) == 0
    assert isinstance(observation, np.ndarray)
    assert isinstance(info, dict)


def test_step_buy(trading_env):
    """Test buy action"""
    initial_balance = trading_env.balance
    action = 1  # Buy
    observation, reward, terminated, truncated, info = trading_env.step(action)
    # Check if shares were bought
    assert trading_env.shares_held > 0
    assert trading_env.balance < initial_balance
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(info, dict)


def test_step_sell(trading_env):
    """Test sell action"""
    # First buy some shares
    trading_env.step(1)
    initial_balance = trading_env.balance

    # Then sell
    action = 2  # Sell
    observation, reward, terminated, truncated, info = trading_env.step(action)
    # Check if shares were sold
    assert trading_env.shares_held == 0
    assert trading_env.balance > initial_balance
    assert isinstance(observation, np.ndarray)


def test_step_hold(trading_env):
    """Test hold action"""
    initial_balance = trading_env.balance
    initial_shares = trading_env.shares_held
    action = 0  # Hold
    observation, reward, terminated, truncated, info = trading_env.step(action)
    # Check if state remained unchanged
    assert trading_env.balance == initial_balance
    assert trading_env.shares_held == initial_shares


def test_observation_space(trading_env):
    """Test observation space dimensions"""
    # Get observation through public reset method
    observation, _ = trading_env.reset()
    # Check observation shape
    expected_shape = 4 + len([col for col in trading_env.df.columns if "norm" in col])
    assert observation.shape[0] == expected_shape


def test_episode_end(trading_env):
    """Test if episode ends correctly"""
    # Step until the end of data
    terminated = False
    steps = 0
    while not terminated and steps < len(trading_env.df):
        _, _, terminated, _, _ = trading_env.step(0)  # Always hold
        steps += 1
    assert terminated
    assert trading_env.current_step == len(trading_env.df) - 1
