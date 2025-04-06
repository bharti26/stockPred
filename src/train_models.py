from src.main import train_on_multiple_stocks
from src.dashboard.app import TICKER_MODEL_MAPPING


def train_all_models():
    """Train models for all predefined tickers."""
    # Get unique tickers from the mapping
    tickers = list(set(TICKER_MODEL_MAPPING.keys()))

    # Training parameters
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    episodes_per_stock = 100

    print(f"Training models for {len(tickers)} tickers...")
    print(f"Tickers: {', '.join(tickers)}")

    # Train models
    results = train_on_multiple_stocks(tickers, start_date, end_date, episodes_per_stock)

    # Print results
    print("\nTraining Results:")
    for ticker, (agent, metrics) in results.items():
        print(f"\n{ticker}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    train_all_models()
