from typing import Dict, List, Optional, Any, Union, Callable
# Standard library imports
import copy
import shutil
import threading
import time

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    def __init__(self) -> None:
        """Initialize the TradingVisualizer."""
        self._base_layout: Dict[str, Any] = {
            "width": 1200,
            "height": 800,
            "template": "plotly_dark",
            "showlegend": True,
            "font": {"family": "Arial, sans-serif", "size": 12, "color": "#FFFFFF"},
            "title": {"font": {"size": 24}},
            "xaxis": {"title": "Time", "gridcolor": "#31333F", "showgrid": True},
            "yaxis": {"title": "Price", "gridcolor": "#31333F", "showgrid": True},
            "plot_bgcolor": "#1E1E1E",
            "paper_bgcolor": "#1E1E1E",
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        }
        self._stop_event: Optional[threading.Event] = None
        self._update_thread: Optional[threading.Thread] = None
        self._current_data: Optional[pd.DataFrame] = None
        self._current_portfolio_state: Optional[Dict[str, Any]] = None
        self._last_update_time: Optional[float] = None
        self._fig: Optional[go.Figure] = None
        self._trades: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, float] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.callbacks: List[Callable[[pd.DataFrame], None]] = []

        # Unified color palette
        self.colors: Dict[str, str] = {
            "background": "#1E1E1E",
            "paper": "#2D2D2D",
            "text": "#FFFFFF",
            "grid": "#404040",
            "line": "#00FF00",
            "buy": "#00FF00",
            "sell": "#FF0000",
            "volume": "#888888",
            "sma20": "#00bfff",  # Deep sky blue for short MA
            "sma50": "#ff69b4",  # Hot pink for long MA
            "bb_bands": "rgba(255, 255, 255, 0.3)",  # Semi-transparent white
            "rsi": "#ffd700",  # Gold for RSI
            "macd": "#00ffff",  # Cyan for MACD
            "signal": "#ff4500",  # Orange-red for signal line
            "profit": "#00ff88",  # Bright green for positive metrics
            "loss": "#ff4444",  # Red for negative metrics
            "neutral": "#ffffff",  # White for neutral metrics
        }

    @property
    def layout(self) -> Dict:
        """Get the current layout settings."""
        return self._base_layout.copy()

    @property
    def trades(self) -> List[Dict[str, Any]]:
        """Get the current trades list."""
        return self._trades

    def calculate_performance_metrics(
        self, data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate performance metrics from the training history or trading data.

        Args:
            data: Either a DataFrame with trading data or a dictionary with portfolio values

        Returns:
            Dict[str, float]: Dictionary containing calculated performance metrics
        """
        metrics = {
            "total_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "pnl": 0.0,
        }

        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Calculate metrics from trading data
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
                    # Calculate metrics from training history
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

            # Trade metrics
            if hasattr(self, "trades") and self.trades:
                profitable_trades = len(
                    [
                        t
                        for t in self.trades
                        if (t["action"] == "sell" and t["price"] > t["entry_price"])
                        or (t["action"] == "buy" and t["price"] < t["entry_price"])
                    ]
                )
                total_trades = len(self.trades)
                metrics["win_rate"] = (
                    float((profitable_trades / total_trades * 100)) if total_trades > 0 else 0
                )
                metrics["pnl"] = float(
                    sum(
                        (
                            (t["price"] - t["entry_price"])
                            if t["action"] == "sell"
                            else (t["entry_price"] - t["price"])
                        )
                        for t in self.trades
                    )
                )

        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")

        self.performance_metrics = metrics
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
                line={"color": self.colors["line"]},
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
                line={"color": self.colors["sma20"]},
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            self._get_layout(
                height=800,
                title_text="Training Progress",
                showlegend=True,
            )
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
            df: DataFrame containing trading data
            trades: List of trade dictionaries
            save_path: Optional path to save the plot
        """
        if df.empty:
            print("Warning: Empty DataFrame provided to plot_trading_session")
            return

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_columns):
            print(
                "Warning: Missing required columns. "
                f"Required: {required_columns}, Found: {list(df.columns)}"
            )
            return

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
                x=df.index, y=df["Volume"], name="Volume", marker={"color": self.colors["volume"]}
            ),
            row=2,
            col=1,
        )

        # Add trade markers if available
        if trades:
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
                        marker={
                            "symbol": "triangle-down",
                            "size": 15,
                            "color": self.colors["sell"],
                        },
                    ),
                    row=1,
                    col=1,
                )

        # Update layout
        fig.update_layout(
            self._get_layout(
                height=800, title_text="Trading Session", xaxis_rangeslider_visible=False
            )
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
        if df.empty:
            print("Warning: Empty DataFrame provided to plot_technical_indicators")
            return

        required_columns = ["Close"]
        if not all(col in df.columns for col in required_columns):
            print(
                "Warning: Missing required columns. "
                f"Required: {required_columns}, Found: {list(df.columns)}"
            )
            return

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Price and Moving Averages", "MACD", "RSI"),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25],
        )

        # Plot price
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"], name="Price", line={"color": self.colors["line"]}
            ),
            row=1,
            col=1,
        )

        # Plot moving averages if available
        if "SMA_20" in df.columns:
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

        if "SMA_50" in df.columns:
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

        # Plot Bollinger Bands if available
        if all(col in df.columns for col in ["BB_Upper", "BB_Lower"]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Upper"],
                    name="BB Upper",
                    line={"color": self.colors["bb_bands"]},
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Lower"],
                    name="BB Lower",
                    line={"color": self.colors["bb_bands"]},
                    fill="tonexty",
                ),
                row=1,
                col=1,
            )

        # Plot MACD if available
        if all(col in df.columns for col in ["MACD", "MACD_Signal"]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD"],
                    name="MACD",
                    line={"color": self.colors["macd"]},
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD_Signal"],
                    name="Signal",
                    line={"color": self.colors["signal"]},
                ),
                row=2,
                col=1,
            )

        # Plot RSI if available
        if "RSI" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI"],
                    name="RSI",
                    line={"color": self.colors["rsi"]},
                ),
                row=3,
                col=1,
            )

            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Update layout
        fig.update_layout(
            self._get_layout(
                height=1000,
                title_text="Technical Analysis",
                showlegend=True,
            )
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_metrics(
        self, history: Dict[str, List[float]], save_path: Optional[str] = None, window: int = 30
    ) -> None:
        """Plot portfolio performance metrics.

        Args:
            history: Dictionary containing portfolio values and returns
            save_path: Optional path to save the plot
            window: Rolling window size for metrics
        """
        if not history or "portfolio_values" not in history:
            print("Warning: Empty or invalid history data provided")
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
                line={"color": self.colors["line"]},
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
                line={"color": self.colors["sma20"]},
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
                    line={"color": self.colors["sma50"]},
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            self._get_layout(
                height=800,
                title_text="Portfolio Performance",
                showlegend=True,
            )
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def start_realtime_updates(
        self, data: pd.DataFrame, portfolio_state: Dict[str, Any], update_interval: float = 1.0
    ) -> None:
        """Start real-time updates for visualization.

        Args:
            data: Initial data for visualization
            portfolio_state: Initial portfolio state
            update_interval: Time interval between updates in seconds
        """
        # Stop any existing updates
        self.stop_realtime_updates()

        # Initialize current data and portfolio state
        self._current_data = copy.deepcopy(data)
        self._current_portfolio_state = copy.deepcopy(portfolio_state)
        self._last_update_time = time.time()

        # Start update thread
        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(target=self._update_loop, args=(update_interval,))
        self._update_thread.daemon = True
        self._update_thread.start()

    def stop_realtime_updates(self) -> bool:
        """Stop real-time updates and clean up resources."""
        # Set stop event first
        if self._stop_event is not None:
            self._stop_event.set()

        # Wait for thread to finish
        if self._update_thread is not None and self._update_thread.is_alive():
            try:
                self._update_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error stopping update thread: {e}")

        # Store thread state for testing
        thread_was_alive = self._update_thread is not None and self._update_thread.is_alive()

        # Clear all resources
        self._update_thread = None
        self._stop_event = None
        self._last_update_time = None
        self._current_data = None
        self._current_portfolio_state = None

        # Return thread state for testing
        return not thread_was_alive

    def update_realtime_data(self, new_data: pd.DataFrame) -> None:
        """Update the current data with new data.

        Args:
            new_data: New data to update with
        """
        self._current_data = copy.deepcopy(new_data)
        self._last_update_time = time.time()

    def update_portfolio_state(self, new_state: Dict[str, Any]) -> None:
        """Update the current portfolio state.

        Args:
            new_state: New portfolio state
        """
        self._current_portfolio_state = copy.deepcopy(new_state)

    def _update_portfolio_metrics(self) -> None:
        """Update portfolio performance metrics."""
        if self._current_portfolio_state is not None:
            metrics = self.calculate_performance_metrics
            self._performance_metrics = metrics(self._current_portfolio_state)

    def save_realtime_visualization(self, path: str) -> None:
        """Save the current visualization to a file.

        Args:
            path: Path where the visualization will be saved
        """
        if self._current_data is not None:
            self.update_realtime_data(self._current_data)
            output_file = "realtime_visualization.html"
            shutil.copy(output_file, path)

    def update_colors(self, color_dict: Dict[str, str]) -> None:
        """Update the color scheme.

        Args:
            color_dict: Dictionary containing color settings
        """
        self.colors.update(color_dict)

    def update_layout(self, new_layout: Dict[str, Any]) -> None:
        """Update the layout settings."""
        # Create a deep copy of the base layout
        updated_layout = self._base_layout.copy()

        # Handle special cases for font sizes
        if "title_font_size" in new_layout:
            title_font_size = new_layout.pop("title_font_size")
            updated_layout["title"]["font"]["size"] = title_font_size
        if "axis_font_size" in new_layout:
            font_size = new_layout.pop("axis_font_size")
            updated_layout["xaxis"]["title_font"] = {"size": font_size}
            updated_layout["yaxis"]["title_font"] = {"size": font_size}

        # Update remaining layout properties
        for key, value in new_layout.items():
            is_dict = isinstance(value, dict)
            key_in_layout = key in updated_layout
            is_dict_layout = isinstance(updated_layout.get(key), dict)
            if is_dict and key_in_layout and is_dict_layout:
                updated_layout[key].update(value)
            else:
                updated_layout[key] = value

        self._base_layout = updated_layout

    def _get_layout(self, **kwargs) -> Dict:
        """Get the complete layout with any additional settings.

        Args:
            **kwargs: Additional layout settings to apply.

        Returns:
            Dict: The complete layout configuration.
        """
        layout = self.layout

        # Handle special cases for font sizes in kwargs
        if "title_font_size" in kwargs:
            title_font_size = kwargs.pop("title_font_size")
            layout["title"]["font"]["size"] = title_font_size
        if "axis_font_size" in kwargs:
            font_size = kwargs.pop("axis_font_size")
            layout["xaxis"]["title_font"] = {"size": font_size}
            layout["yaxis"]["title_font"] = {"size": font_size}

        # Update with remaining kwargs
        layout.update(kwargs)
        return layout

    def _update_loop(self, update_interval: float) -> None:
        """Update loop for real-time visualization.

        Args:
            update_interval: Time between updates in seconds
        """
        while True:
            if self._stop_event is not None and self._stop_event.is_set():
                break
            
            try:
                time.sleep(update_interval)
                if self._current_data is not None:
                    self._last_update_time = time.time()
                    self._update_portfolio_metrics()
                    for callback in self.callbacks:
                        callback(self._current_data)
            except Exception as e:
                print(f"Error in update loop: {str(e)}")
                break
