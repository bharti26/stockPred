from typing import Dict, List

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data.realtime_data import RealTimeStockData
from src.models.dqn_agent import DQNAgent
from src.env.trading_env import StockTradingEnv

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    [
        html.H1("Stock Prediction Dashboard"),
        # Control Panel
        html.Div(
            [
                html.Label("Stock Ticker:"),
                dcc.Input(id="ticker-input", type="text", value="AAPL"),
                html.Button("Start Trading", id="start-button", n_clicks=0),
                html.Label("Update Interval (seconds):"),
                dcc.Dropdown(
                    id="interval-dropdown",
                    options=[
                        {"label": "1 second", "value": 1},
                        {"label": "5 seconds", "value": 5},
                        {"label": "10 seconds", "value": 10},
                        {"label": "30 seconds", "value": 30},
                        {"label": "1 minute", "value": 60},
                    ],
                    value=5,
                ),
            ]
        ),
        # Main Content
        html.Div(
            [
                # Left Column - Price Chart
                html.Div(
                    [
                        dcc.Graph(id="price-chart"),
                        dcc.Interval(
                            id="interval-component", interval=5000, n_intervals=0  # in milliseconds
                        ),
                    ]
                ),
                # Right Column - Metrics and Controls
                html.Div(
                    [
                        # Performance Metrics
                        html.Div(id="performance-metrics"),
                        # Trading Controls
                        html.Div(
                            [
                                html.H3("Trading Controls"),
                                html.Button("Buy", id="buy-button", n_clicks=0),
                                html.Button("Sell", id="sell-button", n_clicks=0),
                                html.Button("Hold", id="hold-button", n_clicks=0),
                            ]
                        ),
                        # Trade History
                        html.Div([html.H3("Trade History"), html.Div(id="trade-history")]),
                    ]
                ),
            ]
        ),
    ]
)

# Initialize global variables
data_store = {"ticker": None, "data_source": None, "agent": None, "trades": []}

# Callbacks
@app.callback(
    [
        Output("price-chart", "figure"),
        Output("performance-metrics", "children"),
        Output("trade-history", "children"),
    ],
    [Input("interval-component", "n_intervals"), Input("start-button", "n_clicks")],
    [State("ticker-input", "value"), State("interval-dropdown", "value")],
)
def update_dashboard(n_clicks, ticker):
    if n_clicks == 0:
        return go.Figure(), [], []

    try:
        # Initialize data source if not already done
        if data_store["ticker"] != ticker:
            data_store["ticker"] = ticker
            data_store["data_source"] = RealTimeStockData(ticker)
            data_store["data_source"].start_streaming()

            # Initialize agent
            env = StockTradingEnv(data_store["data_source"].get_latest_data())
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            data_store["agent"] = DQNAgent(state_dim, action_dim)

        # Get latest data
        df = data_store["data_source"].get_latest_data()
        if df is None or df.empty:
            return go.Figure(), [], []

        # Create price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Price"))

        # Add trade markers
        for trade in data_store["trades"]:
            fig.add_trace(
                go.Scatter(
                    x=[trade["time"]],
                    y=[trade["price"]],
                    mode="markers",
                    marker={
                        "symbol": 'triangle-up' if trade['action'] == 'buy' else 'triangle-down',
                        "size": 10,
                        "color": 'green' if trade['action'] == 'buy' else 'red'
                    },
                    name=f"{trade['action'].capitalize()} at {trade['price']:.2f}",
                )
            )

        # Update layout
        fig.update_layout(
            title=f"{ticker} Price Chart", xaxis_title="Time", yaxis_title="Price ($)"
        )

        # Create performance metrics
        metrics = create_metrics(df, data_store["trades"])

        # Create trade history
        trade_history = create_trade_history(data_store["trades"])

        return fig, metrics, trade_history

    except Exception as e:
        print(f"Error updating dashboard: {e}")
        return go.Figure(), [], []


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
            html.Div(
                [
                    html.H4("Price Performance"),
                    html.P(f"Current Price: ${latest_price:.2f}"),
                    html.P(f"Price Change: {price_change:+.2f} ({price_change_pct:+.2f}%)"),
                ]
            ),
            html.Div(
                [
                    html.H4("Trading Performance"),
                    html.P(f"Total Trades: {total_trades}"),
                    html.P(f"Win Rate: {win_rate:.1f}%"),
                    html.P(f"Total Profit: ${total_profit:+.2f}"),
                ]
            ),
            html.Div([html.H4("Risk Metrics"), html.P(f"Volatility: {volatility:.2%}")]),
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


if __name__ == "__main__":
    app.run_server(debug=True)
