#!/usr/bin/env python3
"""Main entry point for the StockPred dashboard."""

import argparse
import logging

from src.dashboard.app import app
from src.dashboard.cli import create_dashboard_parser

# Configure logger
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    return create_dashboard_parser().parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
