import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent
from src.utils.visualization import TradingVisualizer


class StockEvaluator:
    """A class to evaluate trading strategies through backtesting and performance analysis.

    Attributes:
        agent (DQNAgent): The trained DQN agen
        env (StockTradingEnv): The trading environmen
        visualizer (TradingVisualizer): Visualization tool for evaluation results
    """

    def __init__(self, env: StockTradingEnv, agent: DQNAgent):
        """Initialize the stock evaluator.

        Args:
            env: The trading environmen
            agent: The trained agent to evaluate
        """
        self.env = env
        self.agent = agent
        self.visualizer = TradingVisualizer()
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = []

    def backtest(self, initial_balance: float = 10000.0) -> Dict[str, float]:
        """Perform backtesting of the trading strategy.

        Args:
            initial_balance: Initial portfolio balance

        Returns:
            Dictionary containing backtest results
        """
        print("Starting backtesting...")

        # Reset environment with initial balance
        observation_tuple = self.env.reset()
        state = observation_tuple[0]  # Extract observation from tuple
        self.trades = []
        self.portfolio_values = [initial_balance]

        while True:
            # Get action from agen
            action = self.agent.act(state)

            # Take action in environmen
            next_state, _, terminated, truncated, _ = self.env.step(action)

            # Record trade if action was taken
            if action != 0:  # 0 is typically 'hold'
                self.trades.append(
                    {
                        "step": len(self.trades),
                        "action": "buy" if action == 1 else "sell",
                        "price": self.env.df["Close"].iloc[self.env.current_step],
                        "balance": self.env.balance,
                        "shares": self.env.shares_held,
                    }
                )

            # Record portfolio value
            current_price = self.env.df["Close"].iloc[self.env.current_step]
            current_value = self.env.balance + self.env.shares_held * current_price
            self.portfolio_values.append(current_value)

            done = terminated or truncated
            if done:
                break

            state = next_state

        # Calculate and visualize results
        results = self._calculate_backtest_metrics()
        self._create_backtest_visualizations()

        return results

    def _calculate_backtest_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive backtest performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        daily_returns = pd.Series(returns)

        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        sortino_ratio = np.sqrt(252) * (
            daily_returns.mean() / daily_returns[daily_returns < 0].std()
        )
        max_drawdown = (
            np.maximum.accumulate(portfolio_values) - portfolio_values
        ) / np.maximum.accumulate(portfolio_values)
        max_drawdown = max_drawdown.max() * 100

        # Trading metrics
        winning_trades = len([t for t in self.trades if t["balance"] > t["price"]])
        total_trades = len(self.trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        metrics = {
            "total_return_pct": total_return,
            "annualized_volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "total_trades": total_trades,
            "final_balance": portfolio_values[-1],
            "final_shares": self.env.shares_held,
        }

        # Print metrics
        print("\nBacktest Results:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Volatility: {volatility * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Final Balance: ${portfolio_values[-1]:.2f}")
        print(f"Final Shares Held: {self.env.shares_held}")

        return metrics

    def _create_backtest_visualizations(self) -> None:
        """Create and save visualizations of backtest results."""
        # Prepare data for portfolio metrics visualization
        history = {"portfolio_values": self.portfolio_values}

        # Create visualizations
        self.visualizer.plot_trading_session(
            df=self.env.df, trades=self.trades, save_path="results/backtest_trading_session.html"
        )

        self.visualizer.plot_portfolio_metrics(
            history=history, save_path="results/backtest_portfolio_metrics.html"
        )

        self.visualizer.plot_technical_indicators(
            df=self.env.df, save_path="results/backtest_technical_analysis.html"
        )

        print("\nBacktest visualizations have been saved:")
        print("- Trading session: results/backtest_trading_session.html")
        print("- Portfolio metrics: results/backtest_portfolio_metrics.html")
        print("- Technical analysis: results/backtest_technical_analysis.html")

    def compare_benchmark(self, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Compare strategy performance against a benchmark.

        Args:
            benchmark_returns: Series of benchmark returns

        Returns:
            Dictionary containing comparison metrics
        """
        strategy_returns = pd.Series(np.diff(self.portfolio_values) / self.portfolio_values[:-1])

        # Calculate comparison metrics
        beta = strategy_returns.cov(benchmark_returns) / benchmark_returns.var()
        alpha = (strategy_returns.mean() - beta * benchmark_returns.mean()) * 252
        tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (strategy_returns.mean() - benchmark_returns.mean()) / tracking_error

        comparison = {
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

        # Print comparison
        print("\nBenchmark Comparison:")
        print(f"Alpha: {alpha:.4f}")
        print(f"Beta: {beta:.2f}")
        print(f"Tracking Error: {tracking_error:.4f}")
        print(f"Information Ratio: {information_ratio:.2f}")

        return comparison
