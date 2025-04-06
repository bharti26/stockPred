import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import TradingVisualizer


@pytest.fixture
def visualizer():
    """Create a TradingVisualizer instance"""
    return TradingVisualizer()


@pytest.fixture
def sample_history():
    """Create sample training history data"""
    return {
        "episode_rewards": [10, 20, 15, 25, 30],
        "portfolio_values": [10000, 10500, 10300, 10800, 11000],
        "episode_lengths": [100, 95, 105, 98, 102],
        "epsilon_values": [1.0, 0.8, 0.6, 0.4, 0.2],
    }


@pytest.fixture
def sample_trades():
    """Create sample trade data"""
    return [
        {"step": 1, "action": "buy", "price": 105, "shares": 10},
        {"step": 3, "action": "sell", "price": 110, "shares": 5},
        {"step": 5, "action": "buy", "price": 108, "shares": 8},
    ]


@pytest.fixture
def sample_data():
    """Create sample price and indicator data"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "Close": np.random.normal(100, 10, 100),
            "Open": np.random.normal(100, 10, 100),
            "High": np.random.normal(105, 10, 100),
            "Low": np.random.normal(95, 10, 100),
            "Volume": np.random.randint(1000, 2000, 100),
            "SMA_20": np.random.normal(100, 5, 100),
            "SMA_50": np.random.normal(100, 3, 100),
            "RSI": np.random.uniform(0, 100, 100),
            "MACD": np.random.normal(0, 1, 100),
            "MACD_Signal": np.random.normal(0, 1, 100),
            "BB_Upper": np.random.normal(110, 5, 100),
            "BB_Lower": np.random.normal(90, 5, 100),
        },
        index=dates,
    )
    return df


def test_plot_training_history(visualizer, sample_history, tmp_path):
    """Test training history plotting"""
    save_path = tmp_path / "training_history.html"
    visualizer.plot_training_history(sample_history, str(save_path))
    assert save_path.exists()


def test_plot_trading_session(visualizer, sample_data, sample_trades, tmp_path):
    """Test trading session plotting"""
    save_path = tmp_path / "trading_session.html"
    visualizer.plot_trading_session(sample_data, sample_trades, str(save_path))
    assert save_path.exists()


def test_plot_technical_indicators(visualizer, sample_data, tmp_path):
    """Test technical indicators plotting"""
    save_path = tmp_path / "technical_indicators.html"
    visualizer.plot_technical_indicators(sample_data, str(save_path))
    assert save_path.exists()


def test_plot_portfolio_metrics(visualizer, sample_history, tmp_path):
    """Test portfolio metrics plotting"""
    save_path = tmp_path / "portfolio_metrics.html"
    visualizer.plot_portfolio_metrics(sample_history, str(save_path))
    assert save_path.exists()


def test_visualization_no_save(visualizer, sample_data, sample_history, sample_trades):
    """Test plotting without saving"""
    # These should not raise any errors
    visualizer.plot_training_history(sample_history)
    visualizer.plot_trading_session(sample_data, sample_trades)
    visualizer.plot_technical_indicators(sample_data)
    visualizer.plot_portfolio_metrics(sample_history)


def test_custom_colors(visualizer, sample_data, sample_trades):
    """Test custom color settings"""
    custom_colors = {
        "price": "purple",
        "buy": "green",
        "sell": "red",
        "background": "white",
        "grid": "lightgray",
    }

    # Update colors
    visualizer.update_colors(custom_colors)

    # Test plotting with custom colors
    visualizer.plot_trading_session(sample_data, sample_trades)

    # Verify colors were updated
    assert visualizer.colors["price"] == "purple"
    assert visualizer.colors["buy"] == "green"
    assert visualizer.colors["sell"] == "red"


def test_custom_layout(visualizer, sample_data):
    """Test custom layout settings"""
    custom_layout = {
        "width": 1200,
        "height": 800,
        "title_font_size": 20,
        "axis_font_size": 14,
    }

    # Update layout
    visualizer.update_layout(custom_layout)

    # Test plotting with custom layout
    visualizer.plot_technical_indicators(sample_data)

    # Verify layout was updated
    assert visualizer.layout["width"] == 1200
    assert visualizer.layout["height"] == 800


def test_performance_metrics(visualizer, sample_history):
    """Test performance metrics calculation"""
    metrics = visualizer.calculate_performance_metrics(sample_history)

    # Verify metrics are calculated correctly
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics

    # Verify metrics are numeric
    assert isinstance(metrics["total_return"], float)
    assert isinstance(metrics["sharpe_ratio"], float)
    assert isinstance(metrics["max_drawdown"], float)
    assert isinstance(metrics["win_rate"], float)


def test_invalid_data_handling(visualizer):
    """Test handling of invalid data"""
    # Test with empty data
    empty_data = pd.DataFrame()
    empty_trades = []
    empty_history = {}

    # These should not raise errors
    visualizer.plot_trading_session(empty_data, empty_trades)
    visualizer.plot_training_history(empty_history)
    visualizer.plot_technical_indicators(empty_data)
    visualizer.plot_portfolio_metrics(empty_history)


def test_missing_columns_handling(visualizer, sample_data):
    """Test handling of missing columns"""
    # Remove some columns
    incomplete_data = sample_data.drop(columns=["SMA_20", "RSI"])

    # This should not raise an error
    visualizer.plot_technical_indicators(incomplete_data)
