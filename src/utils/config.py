"""Configuration settings for the stock trading environment."""

from typing import Dict, Any

# Default visualization configuration
DEFAULT_VISUALIZATION_CONFIG: Dict[str, Any] = {
    "colors": {
        "primary": "#1f77b4",  # Default plotly blue
        "secondary": "#ff7f0e",  # Default plotly orange
        "price": "#1f77b4",
        "buy": "#26a69a",
        "sell": "#ef5350",
        "neutral": "#78909c",
        "line": "#1f77b4",
        "sma20": "#ff9800",
        "sma50": "#2196f3",
        "sma200": "#4caf50",
        "rsi": "#ff5722",
        "macd": "#4caf50",
        "signal": "#f44336",
        "histogram": "#9e9e9e",
        "bb": "#ff9800",  # Added Bollinger Bands color
        "bb_upper": "#ff9800",
        "bb_lower": "#ff9800",
        "bb_middle": "#ff9800",
        "bollinger_upper": "#e91e63",
        "bollinger_lower": "#9c27b0",
        "volume": "#607d8b",
        "background": "#ffffff",
        "grid": "#f0f0f0",
        "text": "#333333",
        "profit": "#2ecc71",
        "loss": "#e74c3c",
    },
    "layout": {
        "title": {"font": {"size": 24, "color": "#333333"}},
        "xaxis": {
            "title": {"font": {"size": 14, "color": "#333333"}},
            "showgrid": True,
            "gridcolor": "#f0f0f0"
        },
        "yaxis": {
            "title": {"font": {"size": 14, "color": "#333333"}},
            "showgrid": True,
            "gridcolor": "#f0f0f0"
        },
        "legend": {"font": {"size": 12, "color": "#333333"}},
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
    },
    "candlestick": {
        "increasing": {"line": {"color": "#2ecc71"}},
        "decreasing": {"line": {"color": "#e74c3c"}},
    },
    "indicators": {
        "rsi": {"color": "#3498db", "overbought": 70, "oversold": 30},
        "macd": {"color": "#9b59b6", "signal_color": "#e74c3c"},
        "bb": {"color": "#ff9800", "std_dev": 2},
    },
    "performance": {
        "portfolio": {"color": "#1f77b4", "line_width": 2},
        "benchmark": {"color": "#ff7f0e", "line_width": 2},
        "sharpe": {"color": "#2ecc71"},
        "max_drawdown": {"color": "#e74c3c"},
    },
} 