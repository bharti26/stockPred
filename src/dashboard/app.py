"""Web dashboard for stock prediction visualization."""

import argparse
import os
import platform
import socket
import traceback
from typing import Dict, List, Optional, Tuple, TypeVar, cast

import dash
import flask
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from gymnasium.spaces import Discrete
from plotly.subplots import make_subplots
from typing_extensions import Protocol
import numpy as np

from src.data.realtime_data import RealTimeStockData
from src.env.trading_env import StockTradingEnv
from src.models.dqn_agent import DQNAgent, Experience
from src.data.top_stocks import get_top_stocks

# Initialize the app with external stylesheets
app = dash.Dash(
    __name__,
    title="StockPred Dashboard",
    external_stylesheets=[],
    # Ensure app is accessible from other hosts
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
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
        "environment": {k: v for k, v in os.environ.items() if k in ["PYTHONPATH", "PATH", "HOME"]},
    }
    return flask.Response(response=flask.json.dumps(info), status=200, mimetype="application/json")


# Define ticker-model mapping
TICKER_MODEL_MAPPING = {
    "AAPL": "dqn_AAPL",  # Apple
    "MSFT": "dqn_MSFT",  # Microsoft
    "GOOGL": "dqn_GOOGL",  # Google
    "AMZN": "dqn_AMZN",  # Amazon
    "META": "dqn_META",  # Meta
    "TSLA": "dqn_TSLA",  # Tesla
    "NVDA": "dqn_NVDA",  # NVIDIA
    "JPM": "dqn_JPM",  # JPMorgan
    "V": "dqn_V",  # Visa
    "WMT": "dqn_WMT",  # Walmart
}

# Define layout
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
                            style={"display": "none"},  # Hide by default
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
        # Main chart
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

    def initialize(self, ticker: str, model_name: Optional[str] = None) -> None:
        """Initialize data store with ticker and model."""
        # Use predefined model for ticker if no model is specified
        if model_name is None and ticker in self.ticker_model_mapping:
            model_name = self.ticker_model_mapping[ticker]

        if self.ticker == ticker and self.model_name == model_name:
            return  # Already initialized with these parameters

        # Stop existing data stream if any
        if self.data_source is not None:
            self.data_source.stop_streaming()

        # Initialize new data source
        self.ticker = ticker
        self.model_name = model_name
        initial_data = None  # Initialize initial_data to None
        
        try:
            # Initialize data source
            self.data_source = RealTimeStockData(ticker)

            # Try to fetch initial data
            try:
                initial_data = self.data_source.get_latest_data()
                if initial_data is None or initial_data.empty:
                    raise ValueError(f"Failed to fetch initial data for {ticker}")
            except Exception as e:
                print(f"Warning: Could not fetch initial data: {str(e)}")
                # Continue anyway to allow the dashboard to show error state

            # Start streaming
            try:
                self.data_source.start_streaming()
                print(f"Started streaming {ticker} data at {self.data_source.interval} intervals")
            except Exception as e:
                print(f"Warning: Could not start streaming: {str(e)}")
                # Continue anyway to allow the dashboard to show error state

            # Initialize trading environment and agent
            try:
                # Create models directory if it doesn't exist
                os.makedirs("models", exist_ok=True)

                # Initialize agent with default dimensions if we don't have data yet
                if initial_data is None or initial_data.empty:
                    print("Initializing agent with default dimensions")
                    state_dim = 16  # Default state dimension to match model
                    action_dim = 3  # Default action dimension (buy, sell, hold)
                    self.agent = DQNAgent(state_dim, action_dim)
                else:
                    # Initialize with actual data dimensions
                    env = StockTradingEnv(initial_data)
                    if env.observation_space.shape is None:
                        raise ValueError("Observation space shape is None")
                    state_dim = int(env.observation_space.shape[0])
                    action_space = cast(Discrete, env.action_space)
                    action_dim = int(action_space.n)
                    self.agent = DQNAgent(state_dim, action_dim)

                # Try to load the model if specified
                if model_name:
                    model_path = f"models/{model_name}.pth"
                    if os.path.exists(model_path):
                        try:
                            # Ensure agent dimensions match model dimensions
                            if state_dim != 16:  # Model expects 16 dimensions
                                print("Adjusting agent dimensions to match model")
                                self.agent = DQNAgent(16, action_dim)
                            self.agent.load(model_path)
                            print(f"Loaded model from {model_path}")
                        except Exception as e:
                            print(f"Warning: Failed to load model from {model_path}: {str(e)}")
                            print("Using untrained agent")
                    else:
                        print(f"Model file not found: {model_path}")
                        print("Using untrained agent")
                        # Create a new model file with the current agent state
                        try:
                            self.agent.save(model_path)
                            print(f"Created new model file at {model_path}")
                        except Exception as e:
                            print(f"Warning: Failed to create model file: {str(e)}")
            except Exception as e:
                print(f"Warning: Could not initialize trading environment: {str(e)}")
                # Initialize with default dimensions as fallback
                state_dim = 16  # Default to match model dimensions
                action_dim = 3
                self.agent = DQNAgent(state_dim, action_dim)

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


# Define callback types
T = TypeVar("T")


class DashCallbackProtocol(Protocol):
    """Protocol for Dash callback functions."""

    def __call__(
        self, n_clicks: int, n_intervals: int, ticker: Optional[str], model_name: str
    ) -> Tuple[go.Figure, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
        """Callback function signature."""


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


def update_dashboard(
    n_clicks: int, n_intervals: int, ticker: Optional[str], model_name: Optional[str]
) -> Tuple[go.Figure, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
    """Update dashboard with latest data and predictions."""
    try:
        if ticker is None or not ticker.strip():
            raise ValueError("Please enter a valid ticker symbol")

        # Initialize data store with ticker and optional model
        try:
            data_store.initialize(ticker, model_name)
        except Exception as e:
            error_msg = f"Failed to initialize data store: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        if data_store.data_source is None:
            error_msg = "Data source not initialized"
            print(error_msg)
            return create_error_state(error_msg)

        # Get latest data
        try:
            df = data_store.data_source.get_latest_data()
            if df is None or df.empty:
                error_msg = f"No data available for {ticker}"
                print(error_msg)
                return create_error_state(error_msg)
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        # Create trading environment
        try:
            env = StockTradingEnv(df)
            state = env.reset()
        except Exception as e:
            error_msg = f"Error creating trading environment: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        if data_store.agent is None:
            error_msg = "Trading agent not initialized"
            print(error_msg)
            return create_error_state(error_msg)

        # Get action from agent
        try:
            action = data_store.agent.act(state[0])
        except Exception as e:
            error_msg = f"Error getting agent action: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        # Execute trade
        try:
            next_state, reward, done, info, _ = env.step(action)
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        # Update agent
        try:
            experience = Experience(
                state=state[0], action=action, reward=reward, next_state=next_state[0], done=done
            )
            data_store.agent.remember(
                experience.state,
                experience.action,
                experience.reward,
                experience.next_state,
                experience.done,
            )
            data_store.agent.replay()
        except Exception as e:
            error_msg = f"Error updating agent: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        # Store trade
        try:
            data_store.trades.append(
                {
                    "time": pd.Timestamp.now(),
                    "action": "buy" if action == 1 else "sell",
                    "price": df["Close"].iloc[-1],
                    "shares": 100,
                }
            )
        except Exception as e:
            error_msg = f"Error storing trade: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        # Create visualizations
        try:
            fig = create_chart(df, data_store.trades)
            metrics = create_metrics(df, data_store.trades)
            trade_history = create_trade_history(data_store.trades)
            top_stocks_table = create_top_stocks_table()
        except Exception as e:
            error_msg = f"Error creating visualizations: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return create_error_state(error_msg)

        return (
            fig,
            metrics,
            trade_history,
            "Running",
            {"color": "green", "fontWeight": "bold"},
            top_stocks_table,
        )

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return create_error_state(error_msg)


def create_error_state(
    error_msg: str,
) -> Tuple[go.Figure, List[html.Div], List[html.Div], str, Dict[str, str], html.Div]:
    """Create error state for dashboard."""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text=f"Error: {error_msg}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "red"},
    )

    # Create error message with more details
    error_details = html.Div([
        html.H4("Error Details", style={"color": "red"}),
        html.P(error_msg, style={"color": "red"}),
        html.P("Please try the following:", style={"marginTop": "10px"}),
        html.Ul([
            html.Li("Check if the ticker symbol is correct"),
            html.Li("Verify your internet connection"),
            html.Li("Try again in a few moments"),
            html.Li("Contact support if the issue persists")
        ])
    ])

    return (
        empty_fig,
        [error_details],
        [html.Div([html.P("No trade history available")])],
        "Error",
        {"color": "red", "fontWeight": "bold"},
        html.Div([html.P("No top stocks data available")]),
    )


# Apply the callback with type checking
update_dashboard = cast(DashCallbackProtocol, callback(update_dashboard))


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

    # Add price candlestick chart
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

    # Add volume chart
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

    # Add trade markers
    for trade in trades:
        fig.add_trace(
            go.Scatter(
                x=[trade["time"]],
                y=[trade["price"]],
                mode="markers",
                marker={
                    "symbol": "triangle-up" if trade["action"] == "buy" else "triangle-down",
                    "size": 10,
                    "color": "green" if trade["action"] == "buy" else "red",
                },
                name=f"{trade['action'].capitalize()} at {trade['price']:.2f}",
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)

    return fig


def create_metrics(df: pd.DataFrame, trades: List[Dict]) -> List[html.Div]:
    """Create performance metrics display."""
    if df.empty:
        return [html.Div([html.P("No data available for metrics")])]
    
    try:
        # Calculate basic metrics
        latest_price = df["Close"].iloc[-1]
        price_change = latest_price - df["Close"].iloc[0]
        price_change_pct = (price_change / df["Close"].iloc[0]) * 100
        
        # Calculate trade metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = sum(trade.get("profit", 0) for trade in trades)
        
        # Calculate risk metrics
        returns = df["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Create metric cards
        metrics = [
            html.Div([
                html.H4("Price Performance"),
                html.P(f"Current Price: ${latest_price:.2f}"),
                html.P(f"Price Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")
            ]),
            
            html.Div([
                html.H4("Trading Performance"),
                html.P(f"Total Trades: {total_trades}"),
                html.P(f"Win Rate: {win_rate:.1f}%"),
                html.P(f"Total Profit: ${total_profit:+.2f}")
            ]),
            
            html.Div([
                html.H4("Risk Metrics"),
                html.P(f"Volatility: {volatility:.2%}")
            ])
        ]
        
        return metrics
        
    except Exception as e:
        print(f"Error creating metrics: {e}")
        return [html.Div([html.P("Error calculating metrics")])]


def create_trade_history(trades: List[Dict]) -> List[html.Div]:
    """Create trade history display."""
    if not trades:
        return [html.P("No trades yet")]

    trade_elements = []
    for trade in reversed(trades):  # Show most recent trades first
        trade_elements.append(
            html.Div(
                [
                    html.P(f"Time: {trade['time']}"),
                    html.P(f"Action: {trade['action'].capitalize()}"),
                    html.P(f"Price: ${trade['price']:.2f}"),
                    html.Hr(),
                ]
            )
        )

    return trade_elements


# Add new callback for model selection visibility
@app.callback(
    Output("model-selection-container", "style"),
    [Input("ticker-input", "value")],
)
def update_model_selection_visibility(ticker: str) -> Dict[str, str]:
    """Show/hide model selection based on ticker."""
    if ticker in TICKER_MODEL_MAPPING:
        return {"display": "none"}  # Hide for predefined tickers
    return {"display": "block"}  # Show for custom tickers


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the StockPred Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    app.run_server(host=args.host, port=args.port, debug=args.debug)
