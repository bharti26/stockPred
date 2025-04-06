from typing import Callable, Dict, List, Optional
import threading
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class TradingVisualizer:
    """A unified class for creating interactive visualizations of trading data and performance.

    This class provides methods to create interactive plots using Plotly for:
    - Price and volume data
    - Technical indicators
    - Trading signals
    - Performance metrics
    - Real-time updates
    """

    def __init__(self, update_interval: float = 1.0) -> None:
        """Initialize the visualizer with default settings.

        Args:
            update_interval: Time between visualization updates (seconds)
        """
        # Set default template and color scheme
        pio.templates.default = "plotly_dark"

        # Unified color palette
        self.colors = {
            "background": "#1a1a1a",
            "paper": "#242424",
            "price": "#00ff88",  # Bright green for price
            "volume": "#3498db",  # Blue for volume
            "buy": "#27ae60",  # Green for buy signals
            "sell": "#c0392b",  # Red for sell signals
            "sma20": "#00bfff",  # Deep sky blue for short MA
            "sma50": "#ff69b4",  # Hot pink for long MA
            "bb_bands": "rgba(255, 255, 255, 0.3)",  # Semi-transparent white
            "rsi": "#ffd700",  # Gold for RSI
            "macd": "#00ffff",  # Cyan for MACD
            "signal": "#ff4500",  # Orange-red for signal line
            "grid": "rgba(255, 255, 255, 0.05)",  # Very subtle grid
            "text": "#ffffff",  # White text
            "profit": "#00ff88",  # Bright green for positive metrics
            "loss": "#ff4444",  # Red for negative metrics
            "neutral": "#ffffff",  # White for neutral metrics
        }

        # Initialize layout settings
        self.layout_settings = {
            "font": {
                "family": "Roboto, Arial, sans-serif",
                "size": 14,
                "color": self.colors["text"],
            },
            "title": {
                "font": {
                    "size": 28,
                    "color": self.colors["text"],
                    "family": "Roboto, Arial, sans-serif",
                },
                "y": 0.98,
                "x": 0.5,
                "xanchor": "center",
            },
            "legend": {
                "font": {"size": 12, "color": self.colors["text"]},
                "bgcolor": "rgba(0,0,0,0.3)",
                "bordercolor": "rgba(255,255,255,0.2)",
                "borderwidth": 1,
            },
            "xaxis": {
                "title_font": {"size": 16, "color": self.colors["text"]},
                "tickfont": {"size": 12, "color": self.colors["text"]},
                "gridcolor": self.colors["grid"],
                "showgrid": True,
                "zeroline": False,
            },
            "yaxis": {
                "title_font": {"size": 16, "color": self.colors["text"]},
                "tickfont": {"size": 12, "color": self.colors["text"]},
                "gridcolor": self.colors["grid"],
                "showgrid": True,
                "zeroline": False,
            },
            "plot_bgcolor": self.colors["background"],
            "paper_bgcolor": self.colors["paper"],
            "margin": {"t": 100, "b": 50, "l": 50, "r": 50},
        }

        # Real-time update settings
        self.update_interval = update_interval
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self.trades: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}

    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from the trading data."""
        returns = df["Close"].pct_change()

        # Basic metrics
        total_return = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
        max_drawdown = ((df["Close"] / df["Close"].cummax()) - 1).min() * 100

        # Trade metrics
        if self.trades:
            profitable_trades = len(
                [
                    t
                    for t in self.trades
                    if (t["action"] == "sell" and t["price"] > t["entry_price"])
                    or (t["action"] == "buy" and t["price"] < t["entry_price"])
                ]
            )
            total_trades = len(self.trades)
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

            # Calculate profit/loss
            pnl = sum(
                (
                    (t["price"] - t["entry_price"])
                    if t["action"] == "sell"
                    else (t["entry_price"] - t["price"])
                )
                for t in self.trades
            )
        else:
            win_rate = 0
            pnl = 0

        self.performance_metrics = {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "pnl": pnl,
        }

        return self.performance_metrics

    def plot_training_history(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """Plot training history metrics.

        Args:
            history: Dictionary containing training metrics
            save_path: Optional path to save the plot
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Episode Rewards", "Portfolio Value"),
            vertical_spacing=0.15,
        )

        # Add episode rewards trace
        fig.add_trace(
            go.Scatter(
                y=history["episode_rewards"],
                mode="lines",
                name="Episode Reward",
                line={"color": self.colors["price"]},
            ),
            row=1,
            col=1,
        )

        # Add portfolio value trace
        fig.add_trace(
            go.Scatter(
                y=history["portfolio_values"],
                mode="lines",
                name="Portfolio Value",
                line={"color": self.colors["sma20"]},
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Training Progress",
            showlegend=True,
            **self.layout_settings,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_trading_session(
        self, df: pd.DataFrame, trades: List[Dict], save_path: Optional[str] = None
    ) -> None:
        """Plot a complete trading session with price, volume, and trades.

        Args:
            df: DataFrame with price and indicator data
            trades: List of trade dictionaries
            save_path: Optional path to save the plot
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Price and Trades", "Volume"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker={"color": self.colors["volume"]},
            ),
            row=2,
            col=1,
        )

        # Add trade markers
        buy_points = [t for t in trades if t["action"] == "buy"]
        sell_points = [t for t in trades if t["action"] == "sell"]

        if buy_points:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[t["step"]] for t in buy_points],
                    y=[t["price"] for t in buy_points],
                    mode="markers",
                    name="Buy",
                    marker={"symbol": "triangle-up", "size": 15, "color": self.colors["buy"]},
                ),
                row=1,
                col=1,
            )

        if sell_points:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[t["step"]] for t in sell_points],
                    y=[t["price"] for t in sell_points],
                    mode="markers",
                    name="Sell",
                    marker={"symbol": "triangle-down", "size": 15, "color": self.colors["sell"]},
                ),
                row=1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Trading Session",
            xaxis_rangeslider_visible=False,
            **self.layout_settings,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_technical_indicators(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot technical indicators.

        Args:
            df: DataFrame with price and indicator data
            save_path: Optional path to save the plot
        """
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=("Price and Moving Averages", "RSI", "MACD", "Volume"),
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
        )

        # Price and Moving Averages
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Add SMAs
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                name="SMA 20",
                line={"color": self.colors["sma20"]},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line={"color": self.colors["sma50"]},
            ),
            row=1,
            col=1,
        )

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line={"color": self.colors["bb_bands"], "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line={"color": self.colors["bb_bands"], "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                line={"color": self.colors["rsi"]},
            ),
            row=2,
            col=1,
        )

        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                line={"color": self.colors["macd"]},
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                line={"color": self.colors["signal"]},
            ),
            row=3,
            col=1,
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker={"color": self.colors["volume"]},
            ),
            row=4,
            col=1,
        )

        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Technical Indicators",
            xaxis_rangeslider_visible=False,
            **self.layout_settings,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_metrics(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """Plot portfolio performance metrics.

        Args:
            history: Dictionary containing portfolio metrics
            save_path: Optional path to save the plot
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cumulative Returns",
                "Return Distribution",
                "Drawdown",
                "Rolling Sharpe Ratio",
            ),
        )

        # Calculate metrics
        returns = pd.Series(history["portfolio_values"]).pct_change()
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = pd.Series(history["portfolio_values"]).expanding().max()
        drawdown = pd.Series(history["portfolio_values"]) / rolling_max - 1
        rolling_sharpe = (
            returns.rolling(window=252).mean()
            * np.sqrt(252)
            / (returns.rolling(window=252).std() * np.sqrt(252))
        )

        # Plot cumulative returns
        fig.add_trace(
            go.Scatter(
                y=cumulative_returns,
                name="Cumulative Returns",
                line={"color": self.colors["profit"]},
            ),
            row=1,
            col=1,
        )

        # Plot return distribution
        fig.add_trace(
            go.Histogram(
                x=returns.dropna(),
                name="Return Distribution",
                marker={"color": self.colors["neutral"]},
            ),
            row=1,
            col=2,
        )

        # Plot drawdown
        fig.add_trace(
            go.Scatter(
                y=drawdown,
                name="Drawdown",
                line={"color": self.colors["loss"]},
            ),
            row=2,
            col=1,
        )

        # Plot rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                y=rolling_sharpe,
                name="Rolling Sharpe Ratio",
                line={"color": self.colors["profit"]},
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Portfolio Performance Metrics",
            showlegend=True,
            **self.layout_settings,
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def start_updating(self, data_callback: Callable[[], Optional[pd.DataFrame]]) -> None:
        """Start real-time updates of the visualization.

        Args:
            data_callback: Function that returns the latest data
        """
        if self.is_running:
            return

        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._update, args=(data_callback,), daemon=True
        )
        self.update_thread.start()

    def _update(self, data_callback: Callable[[], Optional[pd.DataFrame]]) -> None:
        """Internal method to handle real-time updates."""
        while self.is_running:
            df = data_callback()
            if df is not None:
                self.plot_trading_session(df, [])  # Empty trades list for real-time updates
            time.sleep(self.update_interval)

    def stop_updating(self) -> None:
        """Stop real-time updates."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
            self.update_thread = None

    def save_html(self, filename: str) -> None:
        """Save the current plot as an HTML file.

        Args:
            filename: Path to save the HTML file
        """
        if hasattr(self, "fig"):
            self.fig.write_html(filename)
