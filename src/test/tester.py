import logging
from typing import Dict, List, Any
import numpy as np

from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent
from src.utils.visualization import TradingVisualizer

# Configure logger
logger = logging.getLogger(__name__)

class Tester:
    """Class for testing trained models."""
    
    def __init__(self, env: StockTradingEnv, agent: DQNAgent) -> None:
        """Initialize tester.
        
        Args:
            env: Trading environment
            agent: Trained DQN agent
        """
        self.env = env
        self.agent = agent
        self.visualizer = TradingVisualizer()
        self.trades: List[Dict[str, Any]] = []

    def test(self, num_episodes: int = 10) -> Dict[str, float]:
        """Test the trained agent.
        
        Args:
            num_episodes: Number of episodes to test
            
        Returns:
            Dictionary containing test metrics
        """
        try:
            metrics = {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
            
            for _ in range(num_episodes):
                state = self.env.reset()
                done = False
                episode_return = 0.0
                
                while not done:
                    action = self.agent.act(state)
                    state, reward, done, _ = self.env.step(action)
                    episode_return += reward
                    
                metrics["total_return"] += episode_return
                
            # Calculate average metrics
            metrics["total_return"] /= num_episodes
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during testing: {str(e)}")
            return {}

    def _create_test_visualizations(self, trades: List[Dict]) -> None:
        """Create and save visualizations of test results.

        Args:
            trades: List of trades executed during testing
        """
        # Plot trading session
        self.visualizer.plot_trading_session(
            data=self.env.df, trades=trades, save_path="results/test_trading_session.html"
        )

        # Plot technical indicators
        self.visualizer.plot_technical_indicators(
            data=self.env.df, save_path="results/test_technical_analysis.html"
        )

        print("\nTest visualizations have been saved:")
        print("- Trading session: results/test_trading_session.html")
        print("- Technical analysis: results/test_technical_analysis.html")

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate various performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        # Calculate daily returns
        daily_returns = self.env.df["Close"].pct_change().dropna()

        # Calculate metrics
        total_return = (self.env.balance / self.env.initial_balance - 1) * 100
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        max_drawdown = (
            (self.env.df["Close"].cummax() - self.env.df["Close"]) / self.env.df["Close"].cummax()
        ).max() * 100

        metrics = {
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "total_trades": len(self.trades),
            "final_balance": self.env.balance,
            "shares_held": self.env.shares_held,
        }

        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Final Balance: ${self.env.balance:.2f}")
        print(f"Shares Held: {self.env.shares_held}")

        return metrics
