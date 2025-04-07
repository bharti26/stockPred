"""Training script for stock trading models."""

import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta

from src.models.dqn_agent import DQNAgent, DQNAgentConfig
from src.env.trading_env import StockTradingEnv
from src.data.stock_data import StockData
from src.dashboard.app import TICKER_MODEL_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_dqn_model(ticker: str = "AAPL", episodes: int = 100):
    """Train a DQN model for stock trading.
    
    Args:
        ticker: Stock symbol to train on
        episodes: Number of training episodes
    """
    try:
        # Get data for the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 years
        
        data_handler = StockData(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        data_handler.fetch_data()
        data_handler.add_technical_indicators()
        data_handler.preprocess_data()
        
        if data_handler.data is None:
            raise ValueError("Failed to fetch and process stock data")
            
        env = StockTradingEnv(data_handler.data)
        
        # Create agent config
        config = DQNAgentConfig(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        agent = DQNAgent(config)
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()
            logger.info(f"Episode {episode + 1}/{episodes} completed")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

def train_all_models() -> Dict[str, Tuple[DQNAgent, Dict[str, float]]]:
    """Train models for all tickers in the mapping."""
    results = {}
    
    for ticker in TICKER_MODEL_MAPPING:
        try:
            # Initialize agent
            agent = DQNAgent(state_size=10, action_size=3)
            
            # Train agent (placeholder for actual training logic)
            metrics = {"total_return": 0.0, "sharpe_ratio": 0.0}
            
            results[ticker] = (agent, metrics)
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            
    return results


if __name__ == "__main__":
    train_dqn_model()
