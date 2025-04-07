"""Web dashboard for stock prediction visualization."""

import argparse
import os
import platform
import traceback
import socket
from typing import Dict, List, Optional, Tuple, Any, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import flask
from dash import Dash, html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from gymnasium.spaces import Discrete
from plotly.subplots import make_subplots
from typing_extensions import Protocol
import logging

from src.data.realtime_data import RealTimeStockData
from src.models.dqn_agent import DQNAgent, DQNAgentConfig
from src.env.trading_env import StockTradingEnv
from src.utils.visualization import TradingVisualizer
from src.data.top_stocks import get_top_stocks
from src.data.stock_data import StockData
from src.utils.cli import create_dashboard_parser
from src.utils.chart_utils import create_candlestick_chart

# Type definitions
T = TypeVar("T")

# Configure logger
logger = logging.getLogger(__name__)

class DashCallbackProtocol(Protocol):
    """Protocol for dash callback functions."""
    def __call__(
        self,
        n_clicks: int,
        n_intervals: int,
        ticker: Optional[str],
        model_name: Optional[str],
    ) -> Tuple[Any, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
        """Protocol for dash callback functions.
        
        Args:
            n_clicks: Number of button clicks
            n_intervals: Number of intervals
            ticker: Stock ticker symbol
            model_name: Name of the trading model
            
        Returns:
            Tuple containing metrics, trade history, status text, status style, and trading status
        """

    def validate_inputs(
        self,
        ticker: Optional[str],
        model_name: Optional[str],
    ) -> bool:
        """Validate input parameters.
        
        Args:
            ticker: Stock ticker symbol
            model_name: Name of the trading model
            
        Returns:
            True if inputs are valid, False otherwise
        """

    def handle_error(
        self,
        error: Exception,
    ) -> Tuple[Any, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
        """Handle errors in callback execution.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Tuple containing error state components
        """


# Initialize the app with external stylesheets
app = Dash(
    __name__,
    title="StockPred Dashboard",
    external_stylesheets=[
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
        "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
    ],
    # Ensure app is accessible from other hosts
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1"
    }],
)

# Add custom CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .top-stocks-table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-family: Arial, sans-serif;
            }
            .top-stocks-table th {
                background-color: #f2f2f2;
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid #ddd;
            }
            .top-stocks-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .top-stocks-table tr:hover {
                background-color: #f5f5f5;
            }
            .positive-change {
                color: #27ae60;
            }
            .negative-change {
                color: #e74c3c;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Set server reference for gunicorn and other WSGI servers
server = app.server


# Add health check route
@server.route("/health")
def health_check() -> Tuple[str, int]:
    """Health check endpoint."""
    return "OK", 200


# Add debug route
@server.route("/debug")
def debug_info() -> flask.Response:
    """Debug information endpoint."""
    info = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "environment": {
            k: v for k, v in os.environ.items()
            if k in ["PYTHONPATH", "PATH", "HOME"]
        }
    }
    return flask.Response(
        response=flask.json.dumps(info),
        status=200,
        mimetype="application/json"
    )


# Define ticker-model mapping
TICKER_MODEL_MAPPING = {
    "AAPL": "dqn_AAPL",  # Apple
    "MSFT": "dqn_MSFT",  # Microsof
    "GOOGL": "dqn_GOOGL",  # Google
    "AMZN": "dqn_AMZN",  # Amazon
    "META": "dqn_META",  # Meta
    "TSLA": "dqn_TSLA",  # Tesla
    "NVDA": "dqn_NVDA",  # NVIDIA
    "JPM": "dqn_JPM",  # JPMorgan
    "V": "dqn_V",  # Visa
    "WMT": "dqn_WMT",  # Walmar
}

# Define layou
app.layout = html.Div(
    [
        # Header with logo and title
        html.Div(
            [
                html.H1("StockPred Dashboard", style={"textAlign": "center"}),
                html.P("Real-time stock prediction visualization", style={"textAlign": "center"}),
            ],
            className="header",
        ),
        # Control panel
        html.Div(
            [
                # Ticker selection
                html.Div(
                    [
                        html.Label("Select Ticker:"),
                        dcc.Dropdown(
                            id="ticker-input",
                            options=[
                                {"label": ticker, "value": ticker}
                                for ticker in TICKER_MODEL_MAPPING
                            ],
                            value="AAPL",
                        ),
                        html.Button("Load Ticker", id="ticker-button", n_clicks=0),
                    ],
                    className="control-item",
                ),
                # Model selection (hidden by default, shown only for custom model selection)
                html.Div(
                    [
                        html.Label("Select Model:"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": f"{ticker} Model", "value": model_name}
                                for ticker, model_name in TICKER_MODEL_MAPPING.items()
                            ],
                            value=None,  # No default value, will be set based on ticker
                            style={"display": "none"},  # Hide by defaul
                        ),
                    ],
                    className="control-item",
                    id="model-selection-container",
                ),
            ],
            className="control-panel",
        ),
        # Top Stocks Section
        html.Div(
            [
                html.H3("Top 10 Stocks"),
                html.Div(id="top-stocks-table"),
            ],
            style={"width": "100%", "margin": "20px 0"},
        ),
        # Main char
        html.Div([dcc.Graph(id="main-chart")], className="chart-container"),
        # Performance metrics
        html.Div(
            [
                html.H3("Performance Metrics"),
                html.Div(id="performance-metrics"),
                html.H3("Trade History"),
                html.Div(id="trade-history"),
            ],
            style={"width": "30%", "display": "inline-block", "vertical-align": "top"},
        ),
        # Footer with status information
        html.Div(
            [
                html.P("Dashboard Status: "),
                html.Span(
                    "Active", id="dashboard-status", style={"color": "green", "fontWeight": "bold"}
                ),
            ],
            style={"textAlign": "center", "marginTop": "20px"},
        ),
        # Update interval
        dcc.Interval(
            id="interval-component", interval=60 * 1000, n_intervals=0  # in milliseconds (1 minute)
        ),
    ],
    className="dash-container",
)


class DataStore:
    """Store for realtime data and trading environment."""

    def __init__(self) -> None:
        self.ticker: Optional[str] = None
        self.model_name: Optional[str] = None
        self.data_source: Optional[RealTimeStockData] = None
        self.agent: Optional[DQNAgent] = None
        self.trades: List[Dict] = []
        self.ticker_model_mapping = TICKER_MODEL_MAPPING

    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Get the latest data from the data source.
        
        Returns:
            Latest data as DataFrame or None if not available
        """
        if self.data_source:
            return self.data_source.get_latest_data()
        return None

    def get_trade_history(self) -> List[Dict]:
        """Get the trade history.
        
        Returns:
            List of trade dictionaries
        """
        return self.trades

    def clear_data(self) -> None:
        """Clear all stored data."""
        self.ticker = None
        self.model_name = None
        self.data_source = None
        self.agent = None
        self.trades = []

    def _initialize_data_source(self, ticker: str) -> Optional[pd.DataFrame]:
        """Initialize data source and fetch initial data."""
        try:
            self.data_source = RealTimeStockData(ticker)
            initial_data = self.data_source.get_latest_data()
            
            if initial_data is None or initial_data.empty:
                print(f"Warning: Failed to fetch initial data for {ticker}")
                return None
                
            try:
                self.data_source.start_streaming()
                print(f"Started streaming {ticker} data at {self.data_source.interval} intervals")
            except Exception as e:
                print(f"Warning: Could not start streaming: {str(e)}")
            
            return initial_data
        except Exception as e:
            print(f"Warning: Could not initialize data source: {str(e)}")
            return None

    def _initialize_agent(self, initial_data: Optional[pd.DataFrame], model_name: Optional[str]) -> None:
        """Initialize trading agent with appropriate dimensions."""
        try:
            os.makedirs("models", exist_ok=True)
            
            # Set default dimensions
            state_dim = 16  # Default state dimension to match model
            action_dim = 3  # Default action dimension (buy, sell, hold)
            
            # Try to get dimensions from data if available
            if initial_data is not None and not initial_data.empty:
                try:
                    env = StockTradingEnv(initial_data)
                    if env.observation_space.shape is not None:
                        state_dim = int(env.observation_space.shape[0])
                        action_space = Discrete(env.action_space)
                        action_dim = int(action_space.n)
                except Exception as e:
                    print(f"Warning: Using default dimensions due to error: {str(e)}")
            
            config = DQNAgentConfig(state_size=state_dim, action_size=action_dim)
            self.agent = DQNAgent(config)
            
            # Load model if specified
            if model_name:
                self._load_or_create_model(model_name, state_dim, action_dim)
                
        except Exception as e:
            print(f"Warning: Could not initialize trading environment: {str(e)}")
            # Initialize with default dimensions as fallback
            config = DQNAgentConfig(state_size=16, action_size=3)
            self.agent = DQNAgent(config)

    def _load_or_create_model(self, model_name: str, state_dim: int, action_dim: int) -> None:
        """Load existing model or create new one."""
        model_path = f"models/{model_name}.pth"
        if os.path.exists(model_path):
            try:
                # Ensure agent dimensions match model dimensions
                if state_dim != 16:  # Model expects 16 dimensions
                    print("Adjusting agent dimensions to match model")
                    config = DQNAgentConfig(state_size=16, action_size=action_dim)
                    self.agent = DQNAgent(config)
                self.agent.load(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}: {str(e)}")
                print("Using untrained agent")
        else:
            print(f"Model file not found: {model_path}")
            print("Using untrained agent")
            try:
                config = DQNAgentConfig(state_size=state_dim, action_size=action_dim)
                self.agent = DQNAgent(config)
            except Exception as e:
                print(f"Warning: Failed to create new agent: {str(e)}")
                config = DQNAgentConfig(state_size=16, action_size=3)
                self.agent = DQNAgent(config)

    def initialize(self, ticker: str, model_name: Optional[str] = None) -> None:
        """Initialize data store with ticker and model."""
        try:
            # Use predefined model for ticker if no model is specified
            if model_name is None and ticker in self.ticker_model_mapping:
                model_name = self.ticker_model_mapping[ticker]

            if self.ticker == ticker and self.model_name == model_name:
                return  # Already initialized with these parameters

            # Stop existing data stream if any
            if self.data_source is not None:
                self.data_source.stop_streaming()

            # Initialize new data source and fetch initial data
            self.ticker = ticker
            self.model_name = model_name
            initial_data = self._initialize_data_source(ticker)
            
            # Initialize trading agent
            self._initialize_agent(initial_data, model_name)
            
            self.trades = []
            
        except Exception as e:
            print(f"Error initializing data store: {e}")
            print(traceback.format_exc())
            # Reset state on error
            self.ticker = None
            self.model_name = None
            self.data_source = None
            self.agent = None
            self.trades = []
            raise


# Create data store
data_store = DataStore()


# Create the callback with proper typing
callback = app.callback(
    [
        Output("main-chart", "figure"),
        Output("performance-metrics", "children"),
        Output("trade-history", "children"),
        Output("dashboard-status", "children"),
        Output("dashboard-status", "style"),
        Output("top-stocks-table", "children"),
    ],
    [Input("ticker-button", "n_clicks"), Input("interval-component", "n_intervals")],
    [State("ticker-input", "value"), State("model-dropdown", "value")],
)


def create_top_stocks_table() -> html.Div:
    """Create a table displaying the top 10 stocks."""
    top_stocks = get_top_stocks()

    if not top_stocks:
        return html.Div([html.P("No top stocks data available")])

    # Create table header
    header = html.Tr(
        [
            html.Th("Symbol"),
            html.Th("Name"),
            html.Th("Price"),
            html.Th("Change"),
            html.Th("Volume"),
        ]
    )

    # Create table rows
    rows = []
    for stock in top_stocks:
        change_class = "positive-change" if stock["change"] >= 0 else "negative-change"
        rows.append(
            html.Tr(
                [
                    html.Td(stock["symbol"]),
                    html.Td(stock["name"]),
                    html.Td(f"${stock['price']:.2f}"),
                    html.Td(
                        f"{stock['change']:.2f} ({stock['change_percent']:.2f}%)",
                        className=change_class,
                    ),
                    html.Td(f"{stock['volume']:,}"),
                ]
            )
        )

    return html.Div([html.Table([header] + rows, className="top-stocks-table")])


def create_metrics(portfolio_state: Dict[str, Any]) -> List[html.Div]:
    """Create metrics display components."""
    if not portfolio_state:
        return []

    metrics = [
        html.Div(f"Portfolio Value: ${portfolio_state.get('portfolio_value', 0):.2f}"),
        html.Div(f"Cash Balance: ${portfolio_state.get('cash_balance', 0):.2f}"),
        html.Div(f"Shares Held: {portfolio_state.get('shares_held', 0)}"),
        html.Div(f"Total Trades: {portfolio_state.get('total_trades', 0)}"),
    ]
    return metrics


def create_trade_history(portfolio_state: Dict[str, Any]) -> List[html.Div]:
    """Create trade history display components."""
    if not portfolio_state:
        return []

    history = []
    for trade in portfolio_state["trades"][-5:]:  # Show last 5 trades
        history.append(
            html.Div(
                f"{trade['time']}: {trade['action']} {trade['shares']} shares at ${trade['price']:.2f}"
            )
        )
    return history


def create_trading_components(latest_data: pd.DataFrame) -> Tuple[StockTradingEnv, DQNAgent]:
    """Create trading environment and agent from latest data.
    
    Args:
        latest_data: Latest stock data
        
    Returns:
        Tuple of trading environment and agent
    """
    env = StockTradingEnv(latest_data)
    state_size = int(env.observation_space.shape[0] if env.observation_space.shape is not None else 0)
    action_size = int(getattr(env.action_space, "n", 0))
    config = DQNAgentConfig(state_size=state_size, action_size=action_size)
    agent = DQNAgent(config)
    return env, agent


def create_portfolio_state(data: RealTimeStockData) -> Dict[str, Any]:
    """Create portfolio state from latest data.

    Args:
        data: RealTimeStockData instance

    Returns:
        Dictionary containing portfolio state
    """
    latest_indicators = data.get_latest_indicators()
    return {
        'portfolio_value': latest_indicators.get('portfolio_value', 0.0),
        'cash_balance': latest_indicators.get('cash_balance', 10000.0),
        'shares_held': latest_indicators.get('shares_held', 0),
        'total_trades': latest_indicators.get('total_trades', 0),
        'latest_price': data.get_latest_price() or 0.0,
    }


def create_error_state() -> Tuple[Any, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
    """Create error state for dashboard.

    Returns:
        Tuple containing error state components
    """
    return (
        go.Figure(),
        [],
        [],
        "Error",
        {"display": "none"},
        html.Div(id="trading-status", children="Error: Failed to update metrics"),
    )


def update_trading_metrics(
    ticker: Optional[str],
    model_name: str,
) -> Tuple[Any, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
    """Update trading metrics and visualizations."""
    if not ticker or not model_name:
        raise PreventUpdate

    try:
        # Initialize data
        data = RealTimeStockData(ticker)
        data.start_streaming()
        latest_data = data.get_latest_data()

        # Create portfolio state
        portfolio_state = create_portfolio_state(data)

        # Create visualizer and update data
        visualizer = TradingVisualizer()
        visualizer.update_realtime_data(latest_data)
        visualizer.update_portfolio_state(portfolio_state)

        # Generate metrics and visualizations
        metrics = create_metrics(portfolio_state)
        trade_history = create_trade_history(portfolio_state)
        
        # Get technical indicators chart with default empty figure
        chart = visualizer.plot_technical_indicators(latest_data)
        if not chart:  # Handle None or empty figure
            chart = go.Figure()

        return (
            chart,
            metrics,
            trade_history,
            f"Trading {ticker} with {model_name}",
            {"display": "block"},
            html.Div(id="trading-status", children="Active"),
        )

    except Exception as e:
        print(f"Error updating trading metrics: {e}")
        return create_error_state()


# Register the callback
update_trading_metrics = app.callback(
    Output("chart", "figure"),
    Output("metrics", "children"),
    Output("trade-history", "children"),
    Output("status", "children"),
    Output("status", "style"),
    Output("trading-status", "children"),
    Input("start-button", "n_clicks"),
    Input("interval-component", "n_intervals"),
    State("ticker-input", "value"),
    State("model-dropdown", "value"),
)(update_trading_metrics)


def create_candlestick_trace(
    data: pd.DataFrame,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Candlestick:
    """Create a candlestick trace from data.
    
    Args:
        data: DataFrame with OHLCV data
        row: Optional row number for subplot
        col: Optional column number for subplot
        
    Returns:
        Candlestick trace
    """
    trace = go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price",
    )
    
    if row is not None and col is not None:
        trace.update(row=row, col=col)
    
    return trace


def create_volume_trace(
    data: pd.DataFrame,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Bar:
    """Create a volume bar trace from data.
    
    Args:
        data: DataFrame with OHLCV data
        row: Optional row number for subplot
        col: Optional column number for subplot
        
    Returns:
        Volume bar trace
    """
    trace = go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="blue",
    )
    
    if row is not None and col is not None:
        trace.update(row=row, col=col)
    
    return trace


def create_trade_marker(trade: Dict[str, Any]) -> go.Scatter:
    """Create a scatter trace for a trade marker.
    
    Args:
        trade: Trade data dictionary
        
    Returns:
        Scatter trace for the trade marker
    """
    return go.Scatter(
        x=[trade["time"]],
        y=[trade["price"]],
        mode="markers",
        marker={
            "symbol": "triangle-up" if trade["action"] == "buy" else "triangle-down",
            "size": 10,
            "color": "green" if trade["action"] == "buy" else "red",
        },
        name=f"{trade['action'].capitalize()} at {trade['price']:.2f}",
    )


def create_chart(df: pd.DataFrame, trades: List[Dict]) -> go.Figure:
    """Create the main chart with price data and indicators."""
    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & Indicators", "Volume"),
    )

    # Create candlestick chart
    trace = create_candlestick_chart(
        data=df,
        name="Price",
        showlegend=True,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    )
    fig.add_trace(trace)

    # Add volume chart
    fig.add_trace(create_volume_trace(df, row=2, col=1))

    # Add trade markers
    for trade in trades:
        marker = create_trade_marker(trade)
        marker.update(row=1, col=1)
        fig.add_trace(marker)

    # Update layout
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)

    return fig


# Add new callback for model selection visibility
@app.callback(
    Output("model-selection-container", "style"),
    [Input("ticker-input", "value")],
)
def update_model_selection_visibility(_) -> Dict[str, str]:
    """Update the visibility of the model selection container."""
    return {"display": "block"}


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series.

    Args:
        prices: Series of prices

    Returns:
        Maximum drawdown as a percentage
    """
    if len(prices) < 2:
        return 0.0

    # Calculate cumulative maximum
    cummax = prices.expanding().max()

    # Calculate drawdown
    drawdown = (prices - cummax) / cummax

    # Return maximum drawdown
    return abs(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns series.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    return sharpe if not np.isnan(sharpe) else 0.0


def update_stock_graph_with_indicators(
    selected_ticker: str,
    selected_interval: str,
    selected_indicators: List[str],
) -> Dict:
    """Update the graph with new data and selected technical indicators."""
    try:
        # Get data
        df = get_stock_data(selected_ticker, selected_interval)
        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No data available",
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                },
            }

        # Create figure
        fig = go.Figure()

        # Add candlestick
        fig.add_trace(create_candlestick_trace(df))

        # Add indicators
        for indicator in selected_indicators:
            if indicator == "RSI" and "RSI" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["RSI"],
                        name="RSI",
                        yaxis="y2",
                    )
                )
            elif indicator == "MACD" and all(col in df.columns for col in ["MACD", "Signal"]):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MACD"],
                        name="MACD",
                        yaxis="y2",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Signal"],
                        name="Signal",
                        yaxis="y2",
                    )
                )
            elif indicator == "BB" and all(col in df.columns for col in ["BB_Upper", "BB_Lower"]):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_Upper"],
                        name="BB Upper",
                        line=dict(dash="dash"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_Lower"],
                        name="BB Lower",
                        line=dict(dash="dash"),
                    )
                )

        # Update layout
        fig.update_layout(
            title=f"{selected_ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(
                title="Indicator",
                overlaying="y",
                side="right",
            ),
            height=800,
            width=1200,
            template="plotly_dark",
        )

        return fig.to_dict()

    except Exception as e:
        logger.error("Error updating graph: %s", str(e))
        return {
            "data": [],
            "layout": {
                "title": "Error loading data",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }


def get_stock_data(ticker: str, interval: str) -> pd.DataFrame:
    """Get stock data for the given ticker and interval."""
    try:
        stock_data = StockData(ticker, interval=interval)
        return stock_data.data
    except Exception as e:
        logger.error("Error getting stock data: %s", str(e))
        return pd.DataFrame()


def get_latest_data() -> pd.DataFrame:
    """Get the latest stock data."""
    try:
        return data_store.get_latest_data()
    except Exception as e:
        logger.error("Error getting latest data: %s", str(e))
        return pd.DataFrame()


def get_trade_history() -> List[Dict[str, Any]]:
    """Get the trade history."""
    try:
        return data_store.get_trade_history()
    except Exception as e:
        logger.error("Error getting trade history: %s", str(e))
        return []


@app.callback(
    Output("stock-graph", "figure"),
    [Input("interval-component", "n_intervals")],
)
def update_graph(_) -> go.Figure:
    """Update the stock graph."""
    try:
        # Get latest data
        data = get_latest_data()
        if data.empty:
            return go.Figure()

        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Add candlestick chart
        fig.add_trace(create_candlestick_trace(data, row=1, col=1))

        # Add volume chart
        fig.add_trace(create_volume_trace(data, row=2, col=1))

        # Update layout
        fig.update_layout(
            title="Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume",
            height=800,
            width=1200,
        )

        return fig

    except Exception as e:
        logger.error("Error updating graph: %s", str(e))
        return go.Figure()


@app.callback(
    Output("trade-history", "data"),
    [Input("interval-component", "n_intervals")],
)
def update_trade_history() -> List[Dict[str, Any]]:
    """Update trade history table."""
    try:
        return get_trade_history()
    except Exception as e:
        logger.error("Error updating trade history: %s", str(e))
        return []


def get_portfolio_value() -> float:
    """Get the current portfolio value."""
    if data_store.data_source is None:
        return 0.0
    try:
        latest_data = data_store.data_source.get_latest_data()
        if latest_data is None or latest_data.empty:
            return 0.0
        portfolio_state = create_portfolio_state(data_store.data_source)
        return portfolio_state.get('portfolio_value', 0.0)
    except Exception as e:
        logger.error("Error getting portfolio value: %s", str(e))
        return 0.0


@app.callback(
    Output("portfolio-value", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_portfolio_value() -> str:
    """Update portfolio value display."""
    try:
        return f"Portfolio Value: ${get_portfolio_value():.2f}"
    except Exception as e:
        logger.error("Error updating portfolio value: %s", str(e))
        return "Error updating portfolio value"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return create_dashboard_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run_server(host=args.host, port=args.port, debug=args.debug)
