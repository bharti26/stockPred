"""Shared utilities for creating charts."""

import plotly.graph_objects as go
import pandas as pd


def create_candlestick_chart(
    data: pd.DataFrame,
    name: str = "Price",
    showlegend: bool = True,
    increasing_line_color: str = "#26a69a",
    decreasing_line_color: str = "#ef5350",
) -> go.Candlestick:
    """Create a candlestick chart trace.
    
    Args:
        data: DataFrame containing OHLC data
        name: Name of the trace
        showlegend: Whether to show in legend
        increasing_line_color: Color for increasing candles
        decreasing_line_color: Color for decreasing candles
        
    Returns:
        Configured Candlestick trace
    """
    return go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name=name,
        showlegend=showlegend,
        increasing_line_color=increasing_line_color,
        decreasing_line_color=decreasing_line_color,
    )
