"""Dashboard package for StockPred visualization."""

from typing import Dict, Any, Optional

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


class DashboardManager:
    """Singleton class to manage the dashboard instance."""
    _instance: Optional[Any] = None

    @classmethod
    def get_instance(cls) -> Any:
        """Get the dashboard instance.

        Returns:
            Any: The dashboard instance.
        """
        if cls._instance is None:
            cls._instance = app
        return cls._instance


def get_dashboard() -> Any:
    """Get the dashboard instance.

    Returns:
        Any: The dashboard instance.
    """
    return DashboardManager.get_instance()


__all__ = ["app", "run_server"]
