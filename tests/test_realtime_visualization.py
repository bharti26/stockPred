import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import TradingVisualizer


@pytest.fixture
def visualizer():
    """Create a TradingVisualizer instance"""
    return TradingVisualizer()


@pytest.fixture
def sample_realtime_data():
    """Create sample real-time price and volume data"""
    dates = [datetime.now() - timedelta(minutes=x) for x in range(60)]
    dates.reverse()

    # Generate sample price data with trend and noise
    price = 100
    prices = []
    volumes = []
    for _ in range(60):
        price *= 1 + np.random.normal(0.0002, 0.02)
        volume = np.random.randint(50000, 500000)
        prices.append(price)
        volumes.append(volume)

    df = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)
    return df


@pytest.fixture
def sample_portfolio_state():
    """Create sample portfolio state data"""
    return {
        "cash": 10000.0,
        "shares": 100,
        "current_price": 105.0,
        "total_value": 20500.0,
        "unrealized_pnl": 500.0,
    }


def test_start_realtime_updates(visualizer, sample_realtime_data, sample_portfolio_state):
    """Test starting real-time updates"""
    # Start real-time updates
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=0.1
    )

    # Verify update thread is running
    assert visualizer._update_thread is not None
    assert visualizer._update_thread.is_alive()

    # Stop updates
    visualizer.stop_realtime_updates()
    assert not visualizer._update_thread.is_alive()


def test_realtime_update_interval(visualizer, sample_realtime_data, sample_portfolio_state):
    """Test real-time update interval"""
    update_interval = 0.1
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=update_interval
    )

    # Wait for a few updates
    time.sleep(update_interval * 3)

    # Stop updates
    visualizer.stop_realtime_updates()

    # Verify updates occurred
    assert visualizer._last_update_time is not None


def test_realtime_data_update(visualizer, sample_realtime_data, sample_portfolio_state):
    """Test updating real-time data"""
    # Start updates
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=0.1
    )

    # Update data
    new_price = sample_realtime_data["Close"].iloc[-1] * 1.01
    new_volume = np.random.randint(50000, 500000)
    new_row = pd.DataFrame(
        {"Close": [new_price], "Volume": [new_volume]},
        index=[datetime.now()]
    )
    updated_data = pd.concat([sample_realtime_data, new_row])

    # Update visualizer
    visualizer.update_realtime_data(updated_data)

    # Stop updates
    visualizer.stop_realtime_updates()

    # Verify data was updated
    assert visualizer._current_data is not None
    assert len(visualizer._current_data) == len(updated_data)


def test_realtime_portfolio_update(visualizer, sample_realtime_data, sample_portfolio_state):
    """Test updating portfolio state"""
    # Start updates
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=0.1
    )

    # Update portfolio state
    new_state = sample_portfolio_state.copy()
    new_state["current_price"] *= 1.01
    new_state["total_value"] = (
        new_state["cash"] + new_state["shares"] * new_state["current_price"]
    )
    new_state["unrealized_pnl"] = (
        new_state["total_value"]
        - (sample_portfolio_state["cash"] + sample_portfolio_state["shares"] * 100)
    )

    # Update visualizer
    visualizer.update_portfolio_state(new_state)

    # Stop updates
    visualizer.stop_realtime_updates()

    # Verify state was updated
    assert visualizer._current_portfolio_state is not None
    assert visualizer._current_portfolio_state["current_price"] == new_state["current_price"]


def test_realtime_visualization_save(
    visualizer, sample_realtime_data, sample_portfolio_state, tmp_path
):
    """Test saving real-time visualization"""
    # Start updates
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=0.1
    )

    # Wait for an update
    time.sleep(0.2)

    # Save visualization
    save_path = tmp_path / "realtime_visualization.html"
    visualizer.save_realtime_visualization(str(save_path))

    # Stop updates
    visualizer.stop_realtime_updates()

    # Verify file was created
    assert save_path.exists()


def test_realtime_visualization_cleanup(visualizer, sample_realtime_data, sample_portfolio_state):
    """Test cleanup of real-time visualization resources"""
    # Start updates
    visualizer.start_realtime_updates(
        sample_realtime_data, sample_portfolio_state, update_interval=0.1
    )

    # Stop updates
    visualizer.stop_realtime_updates()

    # Verify resources are cleaned up
    assert visualizer._update_thread is None
    assert visualizer._stop_event is None
    assert visualizer._current_data is None
    assert visualizer._current_portfolio_state is None
