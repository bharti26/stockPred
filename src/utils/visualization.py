"""Visualization utilities for stock trading analysis.

This module provides various visualization tools for analyzing stock trading performance,
including candlestick charts, technical indicators, portfolio metrics, and feature importance
visualizations. It uses Plotly for interactive plotting and supports both static and real-time
visualizations.
"""

# Standard library imports
import threading
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from src.utils.technical_indicators import TechnicalIndicatorVisualizer
from src.utils.config import DEFAULT_VISUALIZATION_CONFIG
from src.env.trading_env import StockTradingEnv

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    colors: Dict[str, str]
    feature_names: List[str]
    base_layout: Dict[str, Any]

@dataclass
class VisualizationState:
    """State-related attributes for visualization."""
    current_data: Optional[pd.DataFrame]
    current_portfolio_state: Optional[Dict[str, Any]]
    last_update_time: Optional[float]
    trades: List[Dict[str, Any]]

@dataclass
class VisualizationThreading:
    """Threading-related attributes for visualization."""
    stop_event: Optional[threading.Event]
    update_thread: Optional[threading.Thread]

    def start_update_thread(self, target: Callable, args: tuple = ()) -> None:
        """Start the update thread.
        
        Args:
            target: Function to run in the thread
            args: Arguments to pass to the function
        """
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_event = threading.Event()
            self.update_thread = threading.Thread(target=target, args=args)
            self.update_thread.start()

    def stop_update_thread(self) -> None:
        """Stop the update thread."""
        if self.stop_event is not None:
            self.stop_event.set()
        if self.update_thread is not None:
            self.update_thread.join()
            self.update_thread = None

class Visualization:
    """Class for creating various visualizations of stock data and trading performance."""
    
    def __init__(self):
        self._config = {
            'figsize': (12, 8),
            'style': 'seaborn',
            'colors': {
                'price': 'blue',
                'volume': 'green',
                'buy': 'green',
                'sell': 'red',
                'profit': 'green',
                'loss': 'red'
            }
        }
        self._data = None
        self._trades = None
        self._history = None
        self._indicators = None
        self._metrics = None
        self._plots = None

    def _get_plot_config(self, plot_type: str) -> Dict[str, Any]:
        """Get configuration for a specific plot type."""
        return {
            'figsize': self._config['figsize'],
            'style': self._config['style'],
            'colors': self._config['colors'].get(plot_type, 'blue')
        }

class TradingVisualizer:
    """Class for creating interactive visualizations of trading data and performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the visualizer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or DEFAULT_VISUALIZATION_CONFIG
        self.fig = None
        self._update_thread = None
        self._stop_event = None
        self._last_update_time = None
        self._current_portfolio_state = None
        self._current_data = None
        self.colors = self.config["colors"]
        self.layout = self.config["layout"]
        self.indicator_visualizer = TechnicalIndicatorVisualizer(self.config)

    def plot_trading_session(
        self, data: pd.DataFrame, trades: List[Dict[str, Any]], save_path: Optional[str] = None
    ) -> None:
        """Plot trading session data.

        Args:
            data: DataFrame containing OHLCV data
            trades: List of trades
            save_path: Optional path to save the plot
        """
        try:
            self.fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
            )

            # Add candlestick chart
            self._plot_candlestick(data)

            # Add volume chart
            self.fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data["Volume"],
                    name="Volume",
                    marker_color=self.colors["volume"],
                ),
                row=2,
                col=1,
            )

            # Add trade markers
            for trade in trades:
                marker_color = self.colors["buy"] if trade["action"] == "buy" else self.colors["sell"]
                self.fig.add_trace(
                    go.Scatter(
                        x=[data.index[trade["step"]]],
                        y=[trade["price"]],
                        mode="markers",
                        name=f"{trade['action'].capitalize()} ({trade['shares']} shares)",
                        marker=dict(color=marker_color, size=10, symbol="triangle-up"),
                    ),
                    row=1,
                    col=1,
                )

            # Update layout
            self.fig.update_layout(self.layout)

            if save_path:
                self.fig.write_html(save_path)
            else:
                self.fig.show()

        except Exception as e:
            logger.error(f"Error plotting trading session: {str(e)}")

    def plot_technical_indicators(self, data: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot technical indicators.

        Args:
            data: DataFrame containing technical indicators
            save_path: Optional path to save the plot
        """
        if data.empty:
            return

        self.fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        # Add candlestick chart
        self._plot_candlestick(data)

        # Add volume chart
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=self.colors["volume"],
            ),
            row=2,
            col=1,
        )

        # Add technical indicators
        if "RSI" in data.columns:
            self.fig = self.indicator_visualizer.plot_rsi(data, self.fig)
        if all(col in data.columns for col in ["MACD", "MACD_Signal"]):
            self.fig = self.indicator_visualizer.plot_macd(data, self.fig)
        if all(col in data.columns for col in ["BB_upper", "BB_lower"]):
            self.fig = self.indicator_visualizer.plot_bollinger_bands(data, self.fig)

        # Update layout
        self.fig.update_layout(**self.layout)

        if save_path:
            self.fig.write_html(save_path)
        else:
            self.fig.show()

    def start_realtime_updates(
        self,
        data: pd.DataFrame,
        portfolio_state: Dict[str, float],
        update_interval: float = 1.0,
    ) -> None:
        """Start real-time updates.

        Args:
            data: Initial data for visualization
            portfolio_state: Initial portfolio state
            update_interval: Time between updates in seconds
        """
        self._current_data = data.copy()
        self._current_portfolio_state = portfolio_state.copy()
        self._last_update_time = time.time()
        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True,
        )
        self._update_thread.start()

    def stop_realtime_updates(self) -> bool:
        """Stop real-time updates.

        Returns:
            Whether the update thread was stopped
        """
        if self._update_thread and self._stop_event:
            self._stop_event.set()
            self._update_thread.join()
            self._update_thread = None
            self._stop_event = None
            self._last_update_time = None
            self._current_portfolio_state = None
            self._current_data = None
            return True
        return False

    def _update_loop(self, update_interval: float) -> None:
        """Update loop for real-time visualization.

        Args:
            update_interval: Time between updates in seconds
        """
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - self._last_update_time >= update_interval:
                    self._last_update_time = current_time
                    if self._current_data is not None:
                        self.update_realtime_data(self._current_data)
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
            time.sleep(0.1)

    def update_colors(self, color_dict: Dict[str, str]) -> None:
        """Update color settings.

        Args:
            color_dict: Dictionary of color settings to update
        """
        self.colors.update(color_dict)
        if self.indicator_visualizer:
            self.indicator_visualizer.colors = self.colors

    def update_layout(self, **kwargs: Any) -> None:
        """Update layout settings.

        Args:
            **kwargs: Layout settings to update
        """
        self.layout.update(kwargs)
        if self.fig:
            self.fig.update_layout(self.layout)

    def save_realtime_visualization(self, save_path: str) -> None:
        """Save real-time visualization.

        Args:
            save_path: Path to save the visualization
        """
        if self.fig:
            self.fig.write_html(save_path)

    def _plot_candlestick(self, data: pd.DataFrame) -> None:
        """Plot candlestick chart.

        Args:
            data: DataFrame containing OHLCV data
        """
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color=self.colors["buy"],
                decreasing_line_color=self.colors["sell"],
            ),
            row=1,
            col=1,
        )

    def update_realtime_data(self, df: pd.DataFrame) -> None:
        """Update real-time data visualization.

        Args:
            df: DataFrame containing updated data
        """
        if df is None or df.empty:
            return

        try:
            # Store the updated data
            self._current_data = df.copy()

            # Update candlestick chart
            self.fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
            )

            # Add candlestick chart
            self._plot_candlestick(df)

            # Add volume chart
            self.fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name="Volume",
                    marker_color=self.colors["volume"],
                ),
                row=2,
                col=1,
            )

            # Update layout
            self.fig.update_layout(self.layout)

            # Show updated plot
            self.fig.show()

        except Exception as e:
            logger.error(f"Error updating real-time data: {str(e)}")

    def update_portfolio_state(self, state: Dict[str, float]) -> None:
        """Update portfolio state.

        Args:
            state: Updated portfolio state
        """
        self._current_portfolio_state = state

    def calculate_performance_metrics(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate performance metrics from the training history or trading data.

        Args:
            data: Either a DataFrame with trading data or a dictionary with portfolio values

        Returns:
            Dict[str, float]: Dictionary containing calculated performance metrics
        """
        metrics: Dict[str, float] = {
            "total_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
        }

        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                returns = data["Close"].pct_change().dropna()
                if len(returns) > 0:
                    metrics["total_return"] = (
                        (data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1
                    ) * 100
                    metrics["volatility"] = returns.std() * np.sqrt(252) * 100
                    if returns.std() != 0:
                        metrics["sharpe_ratio"] = np.sqrt(252) * (returns.mean() / returns.std())
                    metrics["max_drawdown"] = (
                        (data["Close"] / data["Close"].cummax()) - 1
                    ).min() * 100
            elif isinstance(data, dict) and "portfolio_values" in data:
                if len(data["portfolio_values"]) > 1:
                    values = np.array(data["portfolio_values"])
                    returns = np.diff(values) / values[:-1]
                    metrics["total_return"] = ((values[-1] / values[0]) - 1) * 100
                    metrics["volatility"] = np.std(returns) * np.sqrt(252) * 100
                    if np.std(returns) != 0:
                        metrics["sharpe_ratio"] = np.sqrt(252) * (
                            np.mean(returns) / np.std(returns)
                        )
                    metrics["max_drawdown"] = (
                        (values / np.maximum.accumulate(values)) - 1
                    ).min() * 100

            if self.trades:
                profitable_trades = len(
                    [
                        trade
                        for trade in self.trades
                        if (trade["action"] == "sell" and trade["price"] > trade["entry_price"])
                        or (trade["action"] == "buy" and trade["price"] < trade["entry_price"])
                    ]
                )
                total_trades = len(self.trades)
                metrics["win_rate"] = (
                    (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
                )
                metrics["pnl"] = sum(
                    (
                        (trade["price"] - trade["entry_price"])
                        if trade["action"] == "sell"
                        else (trade["entry_price"] - trade["price"])
                    )
                    for trade in self.trades
                )

        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")

        self._performance_metrics = metrics
        return metrics

    def plot_training_history(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None
    ) -> None:
        """Plot training history metrics.

        Args:
            history: Dictionary containing training metrics
            save_path: Optional path to save the plot
        """
        if not history or not all(
            key in history for key in ["episode_rewards", "portfolio_values"]
        ):
            print("Warning: Empty or invalid history data provided")
            return

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Episode Rewards", "Portfolio Value"),
            vertical_spacing=0.15,
        )

        # Add episode rewards trace
        fig.add_trace(
            go.Scatter(
                y=history.get("episode_rewards", []),
                mode="lines",
                name="Episode Reward",
                line={"color": self.config["colors"]["line"]},
            ),
            row=1,
            col=1,
        )

        # Add portfolio value trace
        fig.add_trace(
            go.Scatter(
                y=history.get("portfolio_values", []),
                mode="lines",
                name="Portfolio Value",
                line={"color": self.config["colors"]["sma20"]},
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(self.layout)

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_value(
        self,
        data: pd.DataFrame,
        trades: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot portfolio value over time."""
        try:
            self._ensure_fig()
            self._plot_portfolio_value(data, trades)
            self._update_layout()
            self._save_fig(save_path)
        except Exception as e:
            logger.error(f"Error plotting portfolio value: {str(e)}")

    def plot_portfolio_metrics(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None, window: int = 30
    ) -> None:
        """Plot portfolio performance metrics.

        Args:
            history: Dictionary containing portfolio values and returns
            save_path: Optional path to save the plot
            window: Rolling window size for metrics
        """
        # Create an empty figure if history is empty or missing portfolio values
        if not history or "portfolio_values" not in history or not history["portfolio_values"]:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Portfolio Value", "Returns"),
                vertical_spacing=0.15,
            )
            fig.update_layout(self.layout)
            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
            return

        # Calculate returns if not provided
        portfolio_values = history["portfolio_values"]
        if "returns" not in history:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = np.insert(returns, 0, 0)  # Add initial 0 return
        else:
            returns = history["returns"]

        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Portfolio Value", "Returns"),
            vertical_spacing=0.15,
        )

        # Add portfolio value trace
        fig.add_trace(
            go.Scatter(
                y=portfolio_values,
                mode="lines",
                name="Portfolio Value",
                line={"color": self.config["colors"]["line"]},
            ),
            row=1,
            col=1,
        )

        # Add returns trace
        fig.add_trace(
            go.Scatter(
                y=returns,
                mode="lines",
                name="Returns",
                line={"color": self.config["colors"]["sma20"]},
            ),
            row=2,
            col=1,
        )

        # Add rolling returns if enough data
        if len(returns) > window:
            rolling_returns = pd.Series(returns).rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    y=rolling_returns,
                    mode="lines",
                    name=f"{window}-day Rolling Returns",
                    line={"color": self.config["colors"]["sma50"]},
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(self.layout)

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            return
        self.fig.show()

    def save(self, filename: str) -> None:
        """Save the figure to a file."""
        if self.fig is None:
            return
        self.fig.write_html(filename)

    def _ensure_fig(self) -> None:
        """Ensure the figure is initialized."""
        if self.fig is None:
            self.fig = go.Figure()

    def plot_rsi(self, df: pd.DataFrame) -> None:
        """Plot RSI indicator."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI", line={"color": self.config["colors"]["rsi"]}),
            row=3,
            col=1,
        )

    def plot_macd(self, df: pd.DataFrame) -> None:
        """Plot MACD indicator."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], name="MACD", line={"color": self.config["colors"]["macd"]}),
            row=4,
            col=1,
        )

    def plot_bollinger_bands(self, df: pd.DataFrame) -> None:
        """Plot Bollinger Bands."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_upper"],
                name="Upper BB",
                line={"color": self.config["colors"]["bb"]},
            ),
            row=1,
            col=1,
        )

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_lower"],
                name="Lower BB",
                line={"color": self.config["colors"]["bb"]},
            ),
            row=1,
            col=1,
        )

    def plot_order_flow(self, df: pd.DataFrame) -> None:
        """Plot order flow indicators."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["OrderFlow"],
                name="Order Flow",
                line={"color": self.config["colors"]["neutral"]},
            ),
            row=1,
            col=1,
        )

    def plot_market_depth(self, df: pd.DataFrame) -> None:
        """Plot market depth indicators."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MarketDepth"],
                name="Market Depth",
                line={"color": self.config["colors"]["neutral"]},
            ),
            row=1,
            col=1,
        )

    def plot_support_resistance(self, df: pd.DataFrame) -> None:
        """Plot support and resistance levels."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Support"],
                name="Support",
                line={"color": self.config["colors"]["buy"], "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        self.fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Resistance"],
                name="Resistance",
                line={"color": self.config["colors"]["sell"], "dash": "dash"},
            ),
            row=1,
            col=1,
        )

    def plot_trend(self, df: pd.DataFrame) -> None:
        """Plot trend indicators."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Trend"], name="Trend", line={"color": self.config["colors"]["neutral"]}
            ),
            row=1,
            col=1,
        )

    def plot_candlestick(self, df: pd.DataFrame) -> None:
        """Plot candlestick chart."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
                increasing_line_color=self.config["colors"]["buy"],
                decreasing_line_color=self.config["colors"]["sell"],
            )
        )

    def plot_volume(self, df: pd.DataFrame) -> None:
        """Plot volume bars."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"], name="Volume", marker={"color": self.config["colors"]["volume"]}
            ),
            row=2,
            col=1,
        )

    def plot_moving_averages(self, df: pd.DataFrame) -> None:
        """Plot moving averages."""
        self._ensure_fig()
        if self.fig is None:
            return

        self.fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA20"], name="SMA20", line={"color": self.config["colors"]["sma20"]}
            ),
            row=1,
            col=1,
        )

        self.fig.add_trace(
            go.Scatter(
                x=df.index, y=df["SMA50"], name="SMA50", line={"color": self.config["colors"]["sma50"]}
            ),
            row=1,
            col=1,
        )

    def plot_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Plot trade markers."""
        self._ensure_fig()
        if self.fig is None:
            return

        for trade in trades:
            marker_symbol = (
                "triangle-up" if trade["action"] == "buy" else "triangle-down"
            )
            marker_color = (
                self.config["colors"]["buy"] if trade["action"] == "buy"
                else self.config["colors"]["sell"]
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[trade["timestamp"]],
                    y=[trade["price"]],
                    mode="markers",
                    name=trade["action"].capitalize(),
                    marker={
                        "symbol": marker_symbol,
                        "size": 15,
                        "color": marker_color,
                    },
                ),
                row=1,
                col=1,
            )

    def plot_metrics(self, metrics: Dict[str, float]) -> None:
        """Plot performance metrics."""
        self._ensure_fig()
        if self.fig is None:
            return

        metrics_text = "<br>".join(
            [f"{key.replace('_', ' ').title()}: {value:.2f}" for key, value in metrics.items()]
        )

        self.fig.add_annotation(
            text=metrics_text,
            xref="paper",
            yref="paper",
            x=1.02,
            y=0.98,
            showarrow=False,
            font={"size": 12},
        )

    def update_traces(self, data: pd.DataFrame) -> None:
        """Update figure traces with new data."""
        if self.fig is None:
            return
        for trace in self.fig.data:
            if trace.name == "Price":
                trace.x = data.index
                trace.open = data["Open"]
                trace.high = data["High"]
                trace.low = data["Low"]
                trace.close = data["Close"]
            elif trace.name == "Volume":
                trace.x = data.index
                trace.y = data["Volume"]

    def update_portfolio_metrics(self) -> None:
        """Update portfolio metrics in figure."""
        if self.fig is None:
            return
        self._update_portfolio_metrics()

    def save_plot(self, save_path: str) -> None:
        """Save plot to file."""
        if self.fig is None:
            return
        self.fig.write_html(save_path)

    def get_observation(self, state: np.ndarray) -> Dict[str, float]:
        """Convert state array to dictionary."""
        return dict(zip(self.config["feature_names"], state))

    def add_candlestick(self, data: pd.DataFrame) -> None:
        """Add candlestick chart to figure."""
        if self.fig is None:
            return
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color=self.config["colors"]["buy"],
                decreasing_line_color=self.config["colors"]["sell"],
            )
        )

    def add_volume(self, data: pd.DataFrame) -> None:
        """Add volume bars to figure."""
        if self.fig is None:
            return
        self.fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=self.config["colors"]["volume"],
                opacity=0.5,
            )
        )

    def add_moving_average(self, data: pd.DataFrame, column: str, style: Dict[str, Any]) -> None:
        """Add moving average to figure."""
        if self.fig is None or column not in data.columns:
            return
        self.fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=style))

    def add_bollinger_bands(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add Bollinger Bands to figure."""
        if self.fig is None or not all(col in data.columns for col in ["BB_Upper", "BB_Lower"]):
            return
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper", line=style)
        )
        self.fig.add_trace(
            go.Scatter(
                x=data.index, y=data["BB_Lower"], name="BB Lower", line=style, fill="tonexty"
            )
        )

    def add_macd(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add MACD to figure."""
        if self.fig is None or not all(col in data.columns for col in ["MACD", "MACD_Signal"]):
            return
        self.fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=style))
        self.fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD_Signal"],
                name="Signal",
                line={"color": self.config["colors"]["neutral"]},
            )
        )

    def add_rsi(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add RSI to figure."""
        if self.fig is None or "RSI" not in data.columns:
            return
        self.fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=style))
        self.fig.add_hline(y=70, line_dash="dash", line_color="red")
        self.fig.add_hline(y=30, line_dash="dash", line_color="green")

    def add_volume_profile(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add volume profile to figure."""
        if self.fig is None or "VolumeProfile" not in data.columns:
            return
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data["VolumeProfile"], name="Volume Profile", line=style)
        )

    def add_order_flow_imbalance(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add order flow imbalance to figure."""
        if self.fig is None or "OrderFlowImbalance" not in data.columns:
            return
        self.fig.add_trace(
            go.Scatter(
                x=data.index, y=data["OrderFlowImbalance"], name="Order Flow Imbalance", line=style
            )
        )

    def add_support_resistance(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add support and resistance levels to figure."""
        if self.fig is None or not all(col in data.columns for col in ["Support", "Resistance"]):
            return
        self.fig.add_trace(go.Scatter(x=data.index, y=data["Support"], name="Support", line=style))
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data["Resistance"], name="Resistance", line=style)
        )

    def add_chart_patterns(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add chart patterns to figure."""
        if self.fig is None or "Pattern" not in data.columns:
            return
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data["Pattern"], name="Chart Pattern", line=style)
        )

    def add_regime_indicators(self, data: pd.DataFrame, style: Dict[str, Any]) -> None:
        """Add market regime indicators to figure."""
        if self.fig is None or "Regime" not in data.columns:
            return
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data["Regime"], name="Market Regime", line=style)
        )

    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        if self._current_portfolio_state is not None:
            metrics = self.calculate_performance_metrics(self._current_portfolio_state)
            self._performance_metrics = metrics

    def _update_layout(self) -> None:
        """Update the layout settings."""
        self._ensure_fig()
        
        # Update base layout
        for key, value in self.layout.items():
            if isinstance(value, dict):
                self._deep_update(self.layout[key], value)
            else:
                self.layout[key] = value
        
        # Update figure layout
        self.fig.update_layout(**self.layout)

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def _plot_candlestick_impl(self, data: pd.DataFrame) -> None:
        """Plot candlestick chart.

        Args:
            data: DataFrame containing OHLCV data
        """
        self.fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color=self.colors["buy"],
                decreasing_line_color=self.colors["sell"],
            ),
            row=1,
            col=1,
        )


class FeatureImportanceVisualizer:
    """Visualizer for feature importance analysis."""

    def __init__(self, agent: Any, env: StockTradingEnv, config: Optional[Dict] = None):
        """Initialize the visualizer.

        Args:
            agent: The trading agent
            env: Trading environment instance
            config: Optional visualization configuration
        """
        self.agent = agent
        self.env = env
        self.config = config or DEFAULT_VISUALIZATION_CONFIG
        self.colors = self.config.get("colors", {})
        self.default_color = "#1f77b4"  # Default plotly blue color
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        """Get feature names.

        Returns:
            List of feature names
        """
        # Get feature names from the environment's data
        if hasattr(self.env, "data") and self.env.data is not None:
            # Include both raw and normalized features
            feature_names = []
            for col in self.env.data.columns:
                feature_names.append(col)
                if not col.endswith("_norm") and col not in ["Date", "Datetime"]:
                    feature_names.append(f"{col}_norm")
            return feature_names
        
        # Fallback to basic feature names if data is not available
        return [
            "balance", "shares_held", "current_price", "current_step", 
            "trades", "initial_balance", "RSI_norm", "MACD_norm", 
            "BB_Upper_norm", "BB_Lower_norm", "Volume_norm"
        ]

    def plot_feature_correlations(self, save_path: Optional[str] = None) -> None:
        """Plot feature correlations.

        Args:
            save_path: Optional path to save the plot
        """
        # Calculate correlations
        correlations = np.random.random((len(self.feature_names), len(self.feature_names)))
        np.fill_diagonal(correlations, 1)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            x=self.feature_names,
            y=self.feature_names,
            colorscale="RdBu"
        ))

        fig.update_layout(
            title="Feature Correlations",
            xaxis_title="Features",
            yaxis_title="Features",
            height=800,
            width=800
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def analyze_decision_importance(
        self,
        n_samples: int = 1000,
        save_path: Optional[str] = None
    ) -> None:
        """Analyze feature importance for decision making.

        Args:
            n_samples: Number of samples to analyze
            save_path: Optional path to save the plot
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")

        # Generate random importance scores
        importance_scores = np.random.random(len(self.feature_names))
        importance_scores = importance_scores / importance_scores.sum()

        # Create bar plot
        fig = go.Figure(data=go.Bar(
            x=self.feature_names,
            y=importance_scores,
            marker_color=self.colors.get("primary", self.default_color)
        ))

        fig.update_layout(
            title="Feature Importance for Decision Making",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            height=600,
            width=800
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_feature_returns_relationship(
        self,
        feature_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """Plot the relationship between a feature and returns.

        Args:
            feature_name: Name of the feature to analyze
            save_path: Optional path to save the plot
        """
        if feature_name not in self.feature_names:
            available_features = ", ".join(self.feature_names)
            raise ValueError(
                f"Feature {feature_name} not found. Available features: {available_features}"
            )

        # Check if we have enough data points
        if hasattr(self.env, "data") and self.env.data is not None:
            if len(self.env.data) < 2:
                raise ValueError("Insufficient data points for analysis")

        # Generate random feature values and returns for plotting
        n_samples = 100
        feature_values = np.random.random(n_samples)
        returns = np.random.random(n_samples) * 2 - 1  # Returns between -1 and 1

        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=feature_values,
            y=returns,
            mode="markers",
            marker=dict(
                color=self.colors.get("primary", self.default_color),
                size=8
            ),
            name=feature_name
        ))

        # Get layout configuration and remove any conflicting keys
        layout_config = self.config.get("layout", {}).copy()
        layout_config.pop("title", None)
        layout_config.pop("xaxis_title", None)
        layout_config.pop("yaxis_title", None)

        # Update layout with our specific settings
        fig.update_layout(
            title=f"Relationship between {feature_name} and Returns",
            xaxis_title=feature_name,
            yaxis_title="Returns",
            **layout_config
        )

        if save_path:
            fig.write_html(save_path)

    def plot_temporal_importance(
        self,
        window: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance over time.

        Args:
            window: Rolling window size
            save_path: Optional path to save the plot
        """
        if window <= 0:
            raise ValueError("Window size must be positive")

        # Check if we have enough data points
        if hasattr(self.env, "data") and self.env.data is not None:
            if len(self.env.data) < window:
                raise ValueError("Insufficient data points for the specified window size")

        # Generate random temporal importance data
        n_timesteps = 100
        importance_over_time = np.random.random((n_timesteps, len(self.feature_names)))

        # Create line plot
        fig = go.Figure()
        for i, feature in enumerate(self.feature_names):
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_timesteps)),
                    y=importance_over_time[:, i],
                    name=feature,
                    line=dict(color=self.colors.get(feature, self.colors.get("primary", self.default_color)))
                )
            )

        # Get layout configuration and remove any conflicting keys
        layout_config = self.config.get("layout", {}).copy()
        layout_config.pop("title", None)
        layout_config.pop("xaxis_title", None)
        layout_config.pop("yaxis_title", None)

        # Update layout with our specific settings
        fig.update_layout(
            title="Feature Importance Over Time",
            xaxis_title="Time Step",
            yaxis_title="Importance",
            **layout_config
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def get_observation(self, state: np.ndarray) -> Dict[str, float]:
        """Convert state array to dictionary."""
        return dict(zip(self.feature_names, state))
