"""Command line interface utilities for the dashboard."""

import argparse

def create_dashboard_parser(
    description: str = "StockPred Dashboard",
    default_host: str = "0.0.0.0",
    default_port: int = 8050,
) -> argparse.ArgumentParser:
    """Create a standardized argument parser for dashboard applications.
    
    Args:
        description: Description of the dashboard application
        default_host: Default host to bind to
        default_port: Default port to listen on
        
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to run the dashboard on (default: {default_port})"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help=f"Host to bind the dashboard to (default: {default_host})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    return parser 