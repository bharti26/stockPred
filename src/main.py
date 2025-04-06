import os
from typing import List, Tuple, Dict
import torch
from dotenv import load_dotenv

from src.data.stock_data import StockData
from src.data.top_stocks import get_top_stocks
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, Experience
from src.train.trainer import StockTrader
from src.eval.evaluator import StockEvaluator
from src.test.tester import StockTester

# Load environment variables from .env file
load_dotenv()


def train_agent_on_stock(
    ticker: str, start_date: str, end_date: str, episodes: int = 100
) -> Tuple[DQNAgent, StockTradingEnv, Dict[str, float]]:
    """Train a DQN agent for a single stock.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Training data start date
        end_date (str): Training data end date
        episodes (int): Number of training episodes

    Returns:
        tuple: Trained agent, environment, and training metrics
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {ticker} using device: {device}")

    # Initialize stock data
    stock_data = StockData(ticker, start_date, end_date)
    df = stock_data.fetch_data()
    df = stock_data.add_technical_indicators()
    df = stock_data.preprocess_data()

    # Initialize environment and agent
    env = StockTradingEnv(df)
    state_size = env.observation_space.shape[0] if env.observation_space.shape is not None else 0
    action_size = getattr(env.action_space, "n", 0)
    agent = DQNAgent(state_size, action_size, device=str(device))

    # Training loop
    batch_size = 32
    rewards_history = []
    portfolio_values = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            agent.remember(experience)

            # Train on batch
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        portfolio_values.append(env.balance + (env.shares_held * df.iloc[-1]['Close']))

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward:.2f}")

    # Calculate training metrics
    metrics = {
        "final_reward": total_reward,
        "average_reward": sum(rewards_history) / len(rewards_history),
        "final_portfolio": portfolio_values[-1],
        "max_portfolio": max(portfolio_values),
        "min_portfolio": min(portfolio_values)
    }

    return agent, env, metrics


def train_on_multiple_stocks(
    tickers: List[str], start_date: str, end_date: str, episodes_per_stock: int = 100
) -> Dict[str, Tuple[DQNAgent, Dict[str, float]]]:
    """Train DQN agents on multiple stocks.

    Args:
        tickers (List[str]): List of stock tickers
        start_date (str): Training data start date
        end_date (str): Training data end date
        episodes_per_stock (int): Number of episodes per stock

    Returns:
        Dict[str, Tuple[DQNAgent, Dict[str, float]]]:
            Dictionary mapping tickers to their trained agents and metrics
    """
    results = {}

    for ticker in tickers:
        try:
            print(f"\nTraining on {ticker}...")
            agent, env, metrics = train_agent_on_stock(
                ticker,
                start_date,
                end_date,
                episodes_per_stock
            )

            # Save the model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/dqn_{ticker}.pth"
            agent.save(model_path)
            print(f"Model saved to {model_path}")

            results[ticker] = (agent, metrics)

            # Print metrics
            print(f"\nTraining metrics for {ticker}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

        except Exception as e:
            print(f"Error training on {ticker}: {e}")
            continue

    return results


def train_agent(stock_data, n_episodes=1000, batch_size=32, gamma=0.95):
    """Train a DQN agent on historical stock data.

    Args:
        stock_data: Historical stock data
        n_episodes: Number of training episodes
        batch_size: Size of training batch
        gamma: Discount factor

    Returns:
        Trained DQN agent
    """
    # Create environment
    env = StockTradingEnv(stock_data)

    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Create trainer
    trainer = StockTrader(env, agent)

    # Train agent
    trainer.train(n_episodes=n_episodes, batch_size=batch_size, gamma=gamma)

    return agent


def evaluate_agent(agent, stock_data):
    """Evaluate a trained agent on test data.

    Args:
        agent: Trained DQN agent
        stock_data: Test stock data

    Returns:
        Evaluation metrics
    """
    # Create environment
    env = StockTradingEnv(stock_data)

    # Create evaluator
    evaluator = StockEvaluator(env, agent)

    # Evaluate agent
    metrics = evaluator.evaluate()

    return metrics


def test_agent(agent, stock_data):
    """Test a trained agent on new data.

    Args:
        agent: Trained DQN agent
        stock_data: New stock data for testing

    Returns:
        Test metrics
    """
    # Create environment
    env = StockTradingEnv(stock_data)

    # Create tester
    tester = StockTester(env, agent)

    # Test agent
    metrics = tester.test()

    return metrics


def main():
    """Main function to train and evaluate the trading agent."""
    # Load stock data
    stock_data = StockData()
    train_data = stock_data.get_training_data()
    test_data = stock_data.get_test_data()

    # Train agent
    print("Training agent...")
    agent = train_agent(train_data)

    # Evaluate agent
    print("\nEvaluating agent...")
    eval_metrics = evaluate_agent(agent, test_data)
    print("Evaluation metrics:", eval_metrics)

    # Test agent
    print("\nTesting agent...")
    test_metrics = test_agent(agent, test_data)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    # Get top 10 stocks
    top_stocks = get_top_stocks()
    tickers = [stock["symbol"] for stock in top_stocks]

    # Training parameters
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    episodes_per_stock = 100

    print(f"Training on {len(tickers)} stocks: {', '.join(tickers)}")
    print(f"Training period: {start_date} to {end_date}")
    print(f"Episodes per stock: {episodes_per_stock}")

    # Train on all stocks
    results = train_on_multiple_stocks(tickers, start_date, end_date, episodes_per_stock)

    # Print summary
    print("\nTraining Summary:")
    for ticker, (_, metrics) in results.items():
        print(f"\n{ticker}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

    main()
