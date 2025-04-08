"""Training script for stock trading models."""

import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
import os

from src.models.dqn_agent import DQNAgent, DQNAgentConfig
from src.env.trading_env import StockTradingEnv
from src.data.stock_data import StockData
from src.dashboard.app import TICKER_MODEL_MAPPING

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_dqn_model(ticker: str = "AAPL", episodes: int = 100):
    """Train a DQN model for stock trading.
    
    Args:
        ticker: Stock symbol to train on
        episodes: Number of training episodes
    """
    try:
        logger.info(f"[{ticker}] Starting training process")
        
        # Get data for the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 years
        
        logger.info(f"[{ticker}] Fetching data from {start_date.date()} to {end_date.date()}")
        data_handler = StockData(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        data_handler.fetch_data()
        data_handler.add_technical_indicators()
        data_handler.preprocess_data()
        
        if data_handler.data is None:
            raise ValueError("Failed to fetch and process stock data")
            
        logger.info(f"[{ticker}] Creating trading environment")
        env = StockTradingEnv(data_handler.data)
        
        # Create agent config
        config = DQNAgentConfig(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n
        )
        agent = DQNAgent(config)
        
        logger.info(f"[{ticker}] Starting training with {episodes} episodes")
        for episode in range(episodes):
            state, _ = env.reset()  # Unpack the reset tuple
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)  # Unpack all 5 values
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                agent.replay()
            
            if (episode + 1) % 10 == 0:  # Log every 10 episodes
                logger.info(f"[{ticker}] Episode {episode + 1}/{episodes} completed - Total Reward: {episode_reward:.2f}")
            
        logger.info(f"[{ticker}] Training completed")
            
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/dqn_{ticker}.pth"
        agent.save(model_path)
        logger.info(f"[{ticker}] Saved model to {model_path}")
        
        return agent, {"episodes": episodes, "final_reward": episode_reward}
    except Exception as e:
        logger.error(f"[{ticker}] Error during training: {str(e)}")
        return None, {"error": str(e)}

def train_all_models(episodes: int = 100) -> Dict[str, Tuple[DQNAgent, Dict[str, float]]]:
    """Train models for all tickers in the mapping sequentially.
    
    Args:
        episodes: Number of training episodes per model
        
    Returns:
        Dictionary mapping tickers to their trained agents and metrics
    """
    results = {}
    
    for ticker in TICKER_MODEL_MAPPING:
        try:
            logger.info(f"[{ticker}] Starting training process")
            
            agent, metrics = train_dqn_model(ticker, episodes)
            
            if agent is not None:
                results[ticker] = (agent, metrics)
                logger.info(f"[{ticker}] Successfully completed training")
            else:
                logger.error(f"[{ticker}] Failed to train model")
            
        except Exception as e:
            logger.error(f"[{ticker}] Error during training: {str(e)}")
            
    return results


if __name__ == "__main__":
    # Train all models with default parameters
    train_all_models()
