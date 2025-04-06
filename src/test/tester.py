from typing import Any, Dict, List, Tuple

import numpy as np

from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent
from src.utils.visualization import TradingVisualizer


class StockTester:
    """A class to test and evaluate the trained DQN agent's performance.

    Attributes:
        agent (DQNAgent): The trained DQN agent
        env (StockTradingEnv): The trading environment with test data
        visualizer (TradingVisualizer): Visualization tool for test results
    """

    def __init__(self, env: StockTradingEnv, agent: DQNAgent):
        """Initialize the stock tester.

        Args:
            env: The trading environment with test data
            agent: The trained agent to evaluate
        """
        self.env = env
        self.agent = agent
        self.visualizer = TradingVisualizer()
        self.trades: List[Dict[str, Any]] = []

    def test(self, episodes: int = 1) -> Tuple[float, List[Dict]]:
        """Test the agent's performance on the test dataset.

        Args:
            episodes: Number of test episodes to run

        Returns:
            Tuple containing final portfolio value and list of trades
        """
        print(f"Starting testing for {episodes} episodes...")

        best_portfolio_value = 0
        best_trades = []

        for episode in range(episodes):
            observation_tuple = self.env.reset()
            state = observation_tuple[0]  # Extract observation from tuple
            done = False
            self.trades = []

            while not done:
                # Agent selects action (no exploration during testing)
                action = self.agent.act(state)  # Remove test=True parameter

                # Take action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Record trade if action was taken
                if action != 0:  # 0 is typically 'hold'
                    current_price = self.env.df["Close"].iloc[self.env.current_step]
                    trade = {
                        "step": len(self.trades),
                        "action": "buy" if action == 1 else "sell",
                        "price": current_price,
                    }
                    self.trades.append(trade)

                state = next_state
                done = terminated or truncated

            # Calculate final portfolio value
            final_price = self.env.df["Close"].iloc[-1]
            portfolio_value = self.env.balance + self.env.shares_held * final_price

            # Keep track of best performance
            if portfolio_value > best_portfolio_value:
                best_portfolio_value = portfolio_value
                best_trades = self.trades.copy()

            print(f"Episode {episode + 1}/{episodes}")
            print(f"Final Portfolio Value: ${portfolio_value:.2f}")
            print(f"Number of trades: {len(self.trades)}")
            print("-" * 50)

        # Create visualizations for best performance
        self._create_test_visualizations(best_trades, best_portfolio_value)

        return best_portfolio_value, best_trades

    def _create_test_visualizations(self, trades: List[Dict], final_value: float) -> None:
        """Create and save visualizations of test results.

        Args:
            trades: List of trades executed during testing
            final_value: Final portfolio value
        """
        # Plot trading session
        self.visualizer.plot_trading_session(
            df=self.env.df, trades=trades, save_path="results/test_trading_session.html"
        )

        # Plot technical indicators
        self.visualizer.plot_technical_indicators(
            df=self.env.df, save_path="results/test_technical_analysis.html"
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
