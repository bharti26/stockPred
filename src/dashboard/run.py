#!/usr/bin/env python3
"""Module for running the dashboard application."""

import argparse
import logging.config
import sys
from pathlib import Path

from src.dashboard.app import app
from src.dashboard.fallback_app import run_server

# Ensure proper imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Setup logging
logging_config_path = Path(__file__).resolve().parent.parent.parent / "logging.conf"
if logging_config_path.exists():
    logging.config.fileConfig(logging_config_path)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "dashboard.log"),
        ],
    )

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch StockPred Dashboard")
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard on (default: 8050)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the dashboard on (default: 0.0.0.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open browser")

    return parser.parse_args()


def main() -> None:
    """Main function to run the dashboard."""
    args = parse_args()
    try:
        # Try to run the main dashboard
        app.run(debug=args.debug, host=args.host, port=args.port)
    except ImportError:
        # Fallback to the simpler version if dependencies are missing
        run_server()


if __name__ == "__main__":
    main()
