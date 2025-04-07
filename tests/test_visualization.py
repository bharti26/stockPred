import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Dict, List, Any

from src.utils.visualization import TradingVisualizer
from src.env.trading_env import StockTradingEnv
from src.rl.advanced_rl import AdvancedDQNAgent as DQNAgent
from src.utils.visualization import FeatureImportanceVisualizer


@pytest.fixture
def visualizer() -> TradingVisualizer:
    """Create a TradingVisualizer instance"""
    return TradingVisualizer()


@pytest.fixture
def sample_history() -> Dict[str, List[float]]:
    """Create sample training history data"""
    return {
        "episode_rewards": [10, 20, 15, 25, 30],
        "portfolio_values": [10000, 10500, 10300, 10800, 11000],
        "episode_lengths": [100, 95, 105, 98, 102],
        "epsilon_values": [1.0, 0.8, 0.6, 0.4, 0.2],
    }


@pytest.fixture
def sample_trades() -> List[Dict[str, Any]]:
    """Create sample trade data"""
    return [
        {"step": 1, "action": "buy", "price": 105, "shares": 10},
        {"step": 3, "action": "sell", "price": 110, "shares": 5},
        {"step": 5, "action": "buy", "price": 108, "shares": 8},
    ]


@pytest.fixture
def sample_data() -> pd.DataFrame:
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
            "RSI_norm": np.random.uniform(0, 1, 100),
            "MACD_norm": np.random.uniform(-1, 1, 100),
            "BB_Upper_norm": np.random.uniform(0, 1, 100),
            "BB_Lower_norm": np.random.uniform(0, 1, 100),
            "Volume_norm": np.random.uniform(0, 1, 100),
        },
        index=dates,
    )
    return df


@pytest.fixture
def feature_importance_visualizer(sample_data: pd.DataFrame) -> FeatureImportanceVisualizer:
    """Create a FeatureImportanceVisualizer instance with sample data"""
    env = StockTradingEnv(sample_data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    return FeatureImportanceVisualizer(agent, env)


def test_plot_training_history(
    visualizer: TradingVisualizer, sample_history: Dict[str, List[float]], tmp_path: Path
) -> None:
    """Test training history plotting"""
    save_path = tmp_path / "training_history.html"
    visualizer.plot_training_history(sample_history, str(save_path))
    assert save_path.exists()


def test_plot_trading_session(
    visualizer: TradingVisualizer,
    sample_data: pd.DataFrame,
    sample_trades: List[Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Test trading session plotting"""
    save_path = tmp_path / "trading_session.html"
    visualizer.plot_trading_session(sample_data, sample_trades, str(save_path))
    assert save_path.exists()


def test_plot_technical_indicators(
    visualizer: TradingVisualizer, sample_data: pd.DataFrame, tmp_path: Path
) -> None:
    """Test technical indicators plotting"""
    save_path = tmp_path / "technical_indicators.html"
    visualizer.plot_technical_indicators(sample_data, str(save_path))
    assert save_path.exists()


def test_plot_portfolio_metrics(
    visualizer: TradingVisualizer, sample_history: Dict[str, List[float]], tmp_path: Path
) -> None:
    """Test portfolio metrics plotting"""
    save_path = tmp_path / "portfolio_metrics.html"
    visualizer.plot_portfolio_metrics(sample_history, str(save_path))
    assert save_path.exists()


def test_visualization_no_save(
    visualizer: TradingVisualizer,
    sample_data: pd.DataFrame,
    sample_history: Dict[str, List[float]],
    sample_trades: List[Dict[str, Any]],
) -> None:
    """Test plotting without saving"""
    # These should not raise any errors
    visualizer.plot_training_history(sample_history)
    visualizer.plot_trading_session(sample_data, sample_trades)
    visualizer.plot_technical_indicators(sample_data)
    visualizer.plot_portfolio_metrics(sample_history)


def test_custom_colors(
    visualizer: TradingVisualizer,
    sample_data: pd.DataFrame,
    sample_trades: List[Dict[str, Any]],
) -> None:
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


def test_custom_layout(visualizer: TradingVisualizer, sample_data: pd.DataFrame) -> None:
    """Test custom layout settings"""
    custom_layout = {
        "width": 1200,
        "height": 800,
        "title": {"font": {"size": 20}},
        "xaxis": {"title": {"font": {"size": 14}}},
        "yaxis": {"title": {"font": {"size": 14}}},
    }

    # Update layout
    visualizer.update_layout(**custom_layout)

    # Test plotting with custom layout
    visualizer.plot_technical_indicators(sample_data)

    # Verify layout was updated
    assert visualizer.layout["width"] == 1200
    assert visualizer.layout["height"] == 800
    assert visualizer.layout["title"]["font"]["size"] == 20
    assert visualizer.layout["xaxis"]["title"]["font"]["size"] == 14
    assert visualizer.layout["yaxis"]["title"]["font"]["size"] == 14


def test_performance_metrics(
    visualizer: TradingVisualizer, sample_history: Dict[str, List[float]]
) -> None:
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


def test_invalid_data_handling(visualizer: TradingVisualizer) -> None:
    """Test handling of invalid data"""
    # Test with empty data
    empty_data: pd.DataFrame = pd.DataFrame()
    empty_trades: List[Dict[str, Any]] = []
    empty_history: Dict[str, List[float]] = {}

    # These should not raise errors
    visualizer.plot_trading_session(empty_data, empty_trades)
    visualizer.plot_training_history(empty_history)
    visualizer.plot_technical_indicators(empty_data)
    visualizer.plot_portfolio_metrics(empty_history)


def test_missing_columns_handling(visualizer: TradingVisualizer, sample_data: pd.DataFrame) -> None:
    """Test handling of missing columns"""
    # Remove some columns
    incomplete_data = sample_data.drop(columns=["SMA_20", "RSI"])

    # This should not raise an error
    visualizer.plot_technical_indicators(incomplete_data)


def test_feature_importance_visualizer(feature_importance_visualizer: FeatureImportanceVisualizer):
    """Test the FeatureImportanceVisualizer class."""
    # Test feature names
    feature_names = feature_importance_visualizer._get_feature_names()
    assert len(feature_names) >= 4  # At least base features
    assert "balance" in feature_names  # Changed from "Balance" to "balance"
    assert any("norm" in name for name in feature_names)  # Check for normalized features

    # Test correlation plot
    feature_importance_visualizer.plot_feature_correlations()

    # Test decision importance analysis
    feature_importance_visualizer.analyze_decision_importance(n_samples=10)

    # Test feature returns relationship
    feature_importance_visualizer.plot_feature_returns_relationship("RSI_norm")

    # Test temporal importance
    feature_importance_visualizer.plot_temporal_importance(window=3)  # Small window for test data


def test_feature_importance_error_handling(
    feature_importance_visualizer: FeatureImportanceVisualizer,
):
    """Test error handling in FeatureImportanceVisualizer."""
    # Test with non-existent feature
    with pytest.raises(ValueError, match="Feature .* not found"):
        feature_importance_visualizer.plot_feature_returns_relationship("non_existent_feature")

    # Test with invalid window size
    with pytest.raises(ValueError, match="Window size must be positive"):
        feature_importance_visualizer.plot_temporal_importance(window=0)

    # Test with invalid number of samples
    with pytest.raises(ValueError, match="Number of samples must be positive"):
        feature_importance_visualizer.analyze_decision_importance(n_samples=0)

    # Test with empty environment data
    empty_df = pd.DataFrame(
        {
            "Close": [100],
            "Open": [100],
            "High": [105],
            "Low": [95],
            "Volume": [1000],
            "RSI_norm": [0.5],
        },
        index=[pd.Timestamp("2023-01-01")],
    )
    empty_env = StockTradingEnv(empty_df)
    empty_env.data = empty_df  # Explicitly set the data attribute
    agent = DQNAgent(2, 3)  # Minimal state and action space
    empty_visualizer = FeatureImportanceVisualizer(agent, empty_env)

    # Test with insufficient data points for temporal importance
    with pytest.raises(ValueError, match="Insufficient data points for the specified window size"):
        empty_visualizer.plot_temporal_importance(window=10)  # Use a larger window size than available data


def test_feature_importance_save_plots(
    feature_importance_visualizer: FeatureImportanceVisualizer, tmp_path: Path
):
    """Test saving feature importance plots to files."""
    # Test saving correlation plot
    correlation_path = tmp_path / "feature_correlations.html"
    feature_importance_visualizer.plot_feature_correlations(str(correlation_path))
    assert correlation_path.exists()

    # Test saving decision importance plot
    decision_path = tmp_path / "decision_importance.html"
    feature_importance_visualizer.analyze_decision_importance(
        n_samples=10, save_path=str(decision_path)
    )
    assert decision_path.exists()

    # Test saving feature returns plot
    returns_path = tmp_path / "feature_returns.html"
    feature_importance_visualizer.plot_feature_returns_relationship(
        "RSI_norm", save_path=str(returns_path)
    )
    assert returns_path.exists()

    # Test saving temporal importance plot
    temporal_path = tmp_path / "temporal_importance.html"
    feature_importance_visualizer.plot_temporal_importance(window=3, save_path=str(temporal_path))
    assert temporal_path.exists()
