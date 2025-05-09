import pytest
import numpy as np
import pandas as pd
import torch
from typing import Callable
from pathlib import Path
from src.data.stock_data import StockData
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, DQNAgentConfig


@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """Create a sample stock dataset for benchmarking"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31")
    n_samples = len(dates)

    # Create base data
    data = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, n_samples),
            "High": np.random.uniform(110, 210, n_samples),
            "Low": np.random.uniform(90, 190, n_samples),
            "Close": np.random.uniform(100, 200, n_samples),
            "Volume": np.random.randint(1000000, 10000000, n_samples),
        },
        index=dates,
    )

    # Add normalized features
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        mean = data[col].mean()
        std = data[col].std()
        data[f"{col}_norm"] = (data[col] - mean) / std

    # Add technical indicator normalized features
    data["SMA20_norm"] = np.random.normal(0, 1, n_samples)
    data["SMA50_norm"] = np.random.normal(0, 1, n_samples)
    data["RSI_norm"] = np.random.normal(0, 1, n_samples)
    data["MACD_norm"] = np.random.normal(0, 1, n_samples)
    data["MACD_Signal_norm"] = np.random.normal(0, 1, n_samples)
    data["BB_Upper_norm"] = np.random.normal(0, 1, n_samples)
    data["BB_Lower_norm"] = np.random.normal(0, 1, n_samples)

    return data


@pytest.fixture
def stock_data() -> StockData:
    """Create StockData instance for benchmarking"""
    return StockData("AAPL", "2020-01-01", "2023-12-31")


@pytest.fixture
def trading_env(sample_stock_data: pd.DataFrame) -> StockTradingEnv:
    """Create trading environment for benchmarking"""
    return StockTradingEnv(sample_stock_data)


@pytest.fixture
def dqn_agent(trading_env: StockTradingEnv) -> DQNAgent:
    """Create DQN agent for benchmarking"""
    state_size = trading_env.observation_space.shape[0] if trading_env.observation_space.shape is not None else 0
    action_size = 3  # buy, sell, hold
    config = DQNAgentConfig(
        state_size=state_size,
        action_size=action_size,
        memory_size=10000,
        batch_size=32,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        device="cpu"
    )
    return DQNAgent(config)


def test_technical_indicators_performance(
    benchmark: Callable, stock_data: StockData, sample_stock_data: pd.DataFrame
) -> None:
    """Benchmark technical indicator calculation performance"""
    stock_data.data = sample_stock_data
    benchmark(stock_data.add_technical_indicators)


def test_data_preprocessing_performance(
    benchmark: Callable, stock_data: StockData, sample_stock_data: pd.DataFrame
) -> None:
    """Benchmark data preprocessing performance"""
    stock_data.data = sample_stock_data
    stock_data.add_technical_indicators()
    benchmark(stock_data.preprocess_data)


def test_env_reset_performance(benchmark: Callable, trading_env: StockTradingEnv) -> None:
    """Benchmark environment reset performance"""

    def reset_env() -> None:
        trading_env.reset()

    benchmark(reset_env)


def test_env_step_performance(benchmark: Callable, trading_env: StockTradingEnv) -> None:
    """Benchmark environment step performance"""
    trading_env.reset()

    def step_action() -> None:
        action = trading_env.action_space.sample()
        trading_env.step(action)

    benchmark(step_action)


def test_agent_act_performance(benchmark: Callable, dqn_agent: DQNAgent) -> None:
    """Benchmark agent action selection performance"""
    state = dqn_agent.reset()

    def select_action() -> None:
        dqn_agent.act(state)

    benchmark(select_action)


def test_agent_remember_performance(benchmark: Callable, dqn_agent: DQNAgent) -> None:
    """Benchmark agent memory storage performance"""
    state = dqn_agent.reset()
    action = dqn_agent.act(state)
    next_state, reward, terminated, truncated, info = dqn_agent.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    def store_experience() -> None:
        dqn_agent.remember(state, action, reward, next_state, terminated)

    benchmark(store_experience)


def test_agent_replay_performance(benchmark: Callable, dqn_agent: DQNAgent) -> None:
    """Benchmark agent replay performance"""
    # Fill replay buffer
    for _ in range(dqn_agent.batch_size):
        state = dqn_agent.reset()
        action = dqn_agent.act(state)
        next_state, reward, terminated, truncated, info = dqn_agent.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        dqn_agent.remember(state, action, reward, next_state, terminated)

    def replay() -> None:
        dqn_agent.replay()

    benchmark(replay)


def test_full_episode_performance(
    benchmark: Callable, trading_env: StockTradingEnv, dqn_agent: DQNAgent
) -> None:
    """Benchmark full episode performance"""

    def run_episode() -> None:
        state, _ = trading_env.reset()
        done = False
        while not done:
            action = dqn_agent.act(state)
            next_state, reward, terminated, truncated, info = trading_env.step(action)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            dqn_agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            done = terminated or truncated
            if len(dqn_agent.memory) >= dqn_agent.batch_size:
                dqn_agent.replay()

    benchmark(run_episode)


def test_target_update_performance(benchmark: Callable, dqn_agent: DQNAgent) -> None:
    """Benchmark target network update performance"""

    def update_target() -> None:
        dqn_agent.update_target_model()

    benchmark(update_target)


def test_save_load_performance(
    benchmark: Callable, dqn_agent: DQNAgent, tmp_path: Path
) -> None:
    """Benchmark model saving and loading performance"""
    save_path = str(tmp_path / "model.pth")

    def save_load() -> None:
        dqn_agent.save(save_path)
        dqn_agent.load(save_path)

    benchmark(save_load)


def test_batch_processing_performance(benchmark: Callable, dqn_agent: DQNAgent) -> None:
    """Benchmark batch processing performance"""
    batch_size = 32
    states = np.random.random((batch_size, dqn_agent.state_size))
    next_states = np.random.random((batch_size, dqn_agent.state_size))

    def process_batch() -> None:
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)

        with torch.no_grad():
            dqn_agent.policy_net(states_tensor)
            dqn_agent.target_net(next_states_tensor)

    benchmark(process_batch)
