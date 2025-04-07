"""Technical indicators visualization utilities.

This module provides visualization tools for common technical indicators used in stock trading,
including RSI, MACD, and Bollinger Bands. It uses Plotly for creating interactive charts
that can be integrated with other visualization components.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

class TechnicalIndicatorVisualizer:
    """Visualizer for technical indicators in stock trading."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the technical indicator visualizer.
        
        Args:
            config: Configuration dictionary containing colors and layout settings
        """
        self.config = config

    def plot_rsi(self, data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Plot Relative Strength Index (RSI) indicator.
        
        Args:
            data: DataFrame containing RSI data
            fig: Plotly figure object to add RSI trace to
            
        Returns:
            Updated Plotly figure object
        """
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                name="RSI",
                line=dict(color=self.config["colors"]["rsi"])
            )
        )
        return fig

    def plot_macd(self, data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Plot Moving Average Convergence Divergence (MACD) indicator.
        
        Args:
            data: DataFrame containing MACD data
            fig: Plotly figure object to add MACD traces to
            
        Returns:
            Updated Plotly figure object
        """
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                name="MACD",
                line=dict(color=self.config["colors"]["macd"])
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD_Signal"],
                name="Signal",
                line=dict(color=self.config["colors"]["macd"], dash="dash")
            )
        )
        return fig

    def plot_bollinger_bands(self, data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        """Plot Bollinger Bands indicator.
        
        Args:
            data: DataFrame containing Bollinger Bands data
            fig: Plotly figure object to add Bollinger Bands traces to
            
        Returns:
            Updated Plotly figure object
        """
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_upper"],
                name="Upper BB",
                line=dict(color=self.config["colors"]["bb"], dash="dash")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["BB_lower"],
                name="Lower BB",
                line=dict(color=self.config["colors"]["bb"], dash="dash")
            )
        )
        return fig
        