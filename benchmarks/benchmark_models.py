#!/usr/bin/env python
"""Benchmark different trading models and strategies."""

import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import torch

from src.data.stock_data import StockData
from src.models.dqn_agent import DQNAgent
from src.env.trading_env import StockTradingEnv


def benchmark_model(
    model_name: str, ticker: str, days: int = 30
) -> Dict[str, Any]:
    """Benchmark a specific model on historical data.

    Args:
        model_name: Name of the model to benchmark
        ticker: Stock ticker to use
        days: Number of days to include in benchmark

    Returns:
        Dictionary with benchmark results
    """
    # Load data
    start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
    end_date = pd.Timestamp.now()

    stock_data = StockData(ticker, start_date.strftime('%Y-%m-%d'),
                           end_date.strftime('%Y-%m-%d'))
    df = stock_data.fetch_data()
    df = stock_data.add_technical_indicators()
    df = stock_data.preprocess_data()

    # Create environment
    env = StockTradingEnv(df)

    # Setup agent based on model name
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.load(f"models/{model_name}.pth")

    # Run benchmark
    start_time = time.time()
    observation, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.act(observation)
        observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

    end_time = time.time()

    # Calculate metrics
    execution_time = end_time - start_time
    initial_portfolio = 10000  # Default initial balance
    final_portfolio = env.balance + (env.shares_held * df.iloc[-1]['Close'])
    roi = (final_portfolio - initial_portfolio) / initial_portfolio * 100

    return {
        "model": model_name,
        "ticker": ticker,
        "days": days,
        "total_reward": total_reward,
        "steps": steps,
        "execution_time_seconds": execution_time,
        "trades_executed": len(env.trades),
        "final_portfolio_value": final_portfolio,
        "roi_percent": roi
    }


def run_benchmarks(
    models: List[str], tickers: List[str], days: int = 30
) -> pd.DataFrame:
    """Run benchmarks across multiple models and tickers.

    Args:
        models: List of model names to benchmark
        tickers: List of ticker symbols to use
        days: Number of days of data to use

    Returns:
        DataFrame with benchmark results
    """
    results = []

    for model in models:
        for ticker in tickers:
            try:
                result = benchmark_model(model, ticker, days)
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {model} on {ticker}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trading models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["dqn_default"],
        help="Model names"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=["AAPL", "MSFT", "GOOG"],
        help="Ticker symbols"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to include"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_results.csv",
        help="Output CSV file"
    )

    args = parser.parse_args()

    print(f"Running benchmarks for models: {args.models}")
    print(f"Using tickers: {args.tickers}")
    print(f"Testing with {args.days} days of data")

    results = run_benchmarks(args.models, args.tickers, args.days)
    print("\nBenchmark Results:")
    print(results)

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
