"""Dashboard package for StockPred visualization."""

from typing import Any, Dict, Optional

from src.dashboard.app import app
from src.dashboard.fallback_app import run_server

# Version information
__version__ = "0.1.0"

# Package metadata
__metadata__: Dict[str, Any] = {
    "name": "stockpred-dashboard",
    "description": "Interactive web dashboard for stock prediction visualization",
    "author": "StockPred Team",
    "requires": ["dash", "plotly", "pandas", "numpy"],
}

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {"port": 8050, "debug": False, "host": "0.0.0.0"}

# Dashboard instance holder
_dashboard_instance: Optional[Any] = None


def get_dashboard() -> Any:
    """Get the dashboard instance.

    Returns:
        Any: The dashboard instance.
    """
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = app
    return _dashboard_instance


"""Dashboard module for the StockPred project."""


__all__ = ["app", "run_server"]
