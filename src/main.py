import logging
from typing import Dict, List

from dotenv import load_dotenv

from src.data.stock_data import StockData
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, DQNAgentConfig
from src.train.trainer import StockTrader
from src.utils.visualization import TradingVisualizer

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

def train_on_multiple_stocks(
    tickers: List[str],
    model_path: str,
    num_episodes: int = 100,
    initial_balance: float = 10000.0,
) -> Dict[str, float]:
    """Train a DQN agent on multiple stocks.
    
    Args:
        tickers: List of stock tickers to train on
        model_path: Path to save the trained model
        num_episodes: Number of episodes to train for
        initial_balance: Initial balance for trading
        
    Returns:
        Dictionary containing training metrics
    """
    try:
        # Initialize data handler for first ticker
        stock_data = StockData(tickers[0], "2020-01-01", "2023-12-31")
        stock_data.fetch_data()
        stock_data.add_technical_indicators()
        stock_data.preprocess_data()
        
        if stock_data.data is None:
            raise ValueError("Failed to fetch and process stock data")
            
        # Initialize environment and agent
        env = StockTradingEnv(stock_data.data, initial_balance=initial_balance)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create agent config
        config = DQNAgentConfig(
            state_size=state_size,
            action_size=action_size
        )
        agent = DQNAgent(config)
        
        # Initialize trainer
        trainer = StockTrader(env, agent)
        
        # Train on each stock
        for ticker in tickers[1:]:  # Skip first ticker as we already have its data
            stock_data = StockData(ticker, "2020-01-01", "2023-12-31")
            stock_data.fetch_data()
            stock_data.add_technical_indicators()
            stock_data.preprocess_data()
            
            if stock_data.data is None:
                logger.warning(f"Failed to fetch data for {ticker}, skipping...")
                continue
                
            env = StockTradingEnv(stock_data.data, initial_balance=initial_balance)
            trainer.env = env
            trainer.train(num_episodes)
            
        # Save the trained model
        agent.save(model_path)
        
        return trainer.get_metrics()
        
    except Exception as e:
        logger.error(f"Error training on multiple stocks: {str(e)}")
        return {}

def main() -> None:
    """Main function to run the stock trading application."""
    try:
        # Initialize components
        stock_data = StockData("AAPL", "2020-01-01", "2023-12-31")
        stock_data.fetch_data()
        stock_data.add_technical_indicators()
        stock_data.preprocess_data()
        
        env = StockTradingEnv(stock_data.data)
        
        # Create agent config
        config = DQNAgentConfig(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        agent = DQNAgent(config)
        
        trader = StockTrader(env, agent)
        visualizer = TradingVisualizer()
        
        # Train the agent
        history = trader.train(episodes=100)
        
        # Visualize results
        visualizer.plot_training_history(history)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
