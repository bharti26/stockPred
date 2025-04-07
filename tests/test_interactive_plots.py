
import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import TradingVisualizer
from tests.utils.test_data import generate_sample_price_data


def generate_sample_trades(df, n_trades=20):
    """Generate sample trade data."""
    trades = []
    dates = df.index.tolist()

    for _ in range(n_trades):
        step = np.random.randint(0, len(dates) - 1)
        action = np.random.choice(["buy", "sell"])
        price = df["Close"].iloc[step]

        trades.append({"step": step, "action": action, "price": price})

    return sorted(trades, key=lambda x: x["step"])


def generate_sample_training_history(n_episodes=100):
    """Generate sample training history."""
    # Generate episode rewards with increasing trend and noise
    rewards = []
    base_reward = -50
    for i in range(n_episodes):
        reward = base_reward + i * 0.5 + np.random.normal(0, 10)
        rewards.append(reward)

    # Generate portfolio values with growth and volatility
    portfolio_values = []
    value = 10000
    for _ in range(n_episodes):
        value *= 1 + np.random.normal(0.01, 0.05)
        portfolio_values.append(value)

    return {"episode_rewards": rewards, "portfolio_values": portfolio_values}


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample data for testing."""
    df = generate_sample_price_data(days=252)
    
    # Add technical indicators
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Calculate Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(window=20).std()

    # Calculate RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def main():
    """Generate sample data and create interactive visualizations."""
    # Create visualizer instance
    visualizer = TradingVisualizer()

    # Generate sample data
    df = generate_sample_price_data(days=252)
    trades = generate_sample_trades(df)
    history = generate_sample_training_history()

    # Create visualizations
    print("Creating interactive visualizations...")

    visualizer.plot_training_history(history=history, save_path="results/interactive_training.html")
    print("✓ Training history visualization saved")

    visualizer.plot_trading_session(
        df=df, trades=trades, save_path="results/interactive_trading.html"
    )
    print("✓ Trading session visualization saved")

    visualizer.plot_technical_indicators(df=df, save_path="results/interactive_technical.html")
    print("✓ Technical indicators visualization saved")

    visualizer.plot_portfolio_metrics(
        history=history, save_path="results/interactive_portfolio.html"
    )
    print("✓ Portfolio metrics visualization saved")

    print("\nAll visualizations have been saved in the 'results' directory.")
    print("Open the HTML files in a web browser to interact with the plots.")


if __name__ == "__main__":
    main()
