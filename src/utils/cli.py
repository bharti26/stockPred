"""Command line argument parsing utilities."""
from argparse import ArgumentParser
from typing import Optional


def create_dashboard_parser(description: Optional[str] = None) -> ArgumentParser:
    """Create a parser for dashboard command line arguments.

    Args:
        description: Optional description for the argument parser.

    Returns:
        ArgumentParser: Configured argument parser with host, port, and debug options.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser

