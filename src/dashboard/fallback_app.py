#!/usr/bin/env python3
"""Fallback version of the dashboard that doesn't require Dash."""

# Standard library imports
import argparse
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

# Third-party imports
import pandas as pd
import yfinance as yf

# Ensure proper imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HTML template
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StockPred Dashboard</title>
    <style>
        body {
            font-family: 'Roboto', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 5px;
        }

        .data-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        table th, table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .up {
            color: #27ae60;
        }

        .down {
            color: #e74c3c;
        }

        .button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }

        .button:hover {
            background-color: #2980b9;
        }

        #ticker-input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        .chart-placeholder {
            width: 100%;
            height: 400px;
            background-color: #f9f9f9;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>StockPred Dashboard</h1>
            <p>Simple stock data visualization</p>
        </div>

        <div class="data-container">
            <div style="display: flex; justify-content: space-between;
                       align-items: center; margin-bottom: 20px;">
                <div>
                    <input type="text" id="ticker-input" value="AAPL"
                           placeholder="Enter ticker symbol">
                    <button class="button" onclick="updateData()">Load Ticker</button>
                </div>
                <div>
                    <span id="last-updated"></span>
                </div>
            </div>

            <div class="chart-placeholder">
                <p>Chart visualization requires Plotly/Dash. This is a simplified view.</p>
            </div>

            <h3>Stock Data</h3>
            <table id="stock-data">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Open</th>
                        <th>High</th>
                        <th>Low</th>
                        <th>Close</th>
                        <th>Volume</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="7" style="text-align: center;">Loading data...</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function updateData() {
            const ticker = document.getElementById('ticker-input').value;
            fetch(`/api/data?ticker=${ticker}`)
                .then(response => response.json())
                .then(data => {
                    // Update last updated time
                    const lastUpdated = 'Last updated: ' + new Date().toLocaleTimeString();
                    document.getElementById('last-updated').textContent = lastUpdated;

                    // Update stock data table
                    const table = document.getElementById('stock-data');
                    const tableBody = table.getElementsByTagName('tbody')[0];
                    tableBody.innerHTML = '';

                    data.forEach(row => {
                        const tr = document.createElement('tr');

                        const dateCell = document.createElement('td');
                        dateCell.textContent = row.date;
                        tr.appendChild(dateCell);

                        const openCell = document.createElement('td');
                        openCell.textContent = row.open.toFixed(2);
                        tr.appendChild(openCell);

                        const highCell = document.createElement('td');
                        highCell.textContent = row.high.toFixed(2);
                        tr.appendChild(highCell);

                        const lowCell = document.createElement('td');
                        lowCell.textContent = row.low.toFixed(2);
                        tr.appendChild(lowCell);

                        const closeCell = document.createElement('td');
                        closeCell.textContent = row.close.toFixed(2);
                        tr.appendChild(closeCell);

                        const volumeCell = document.createElement('td');
                        volumeCell.textContent = row.volume.toLocaleString();
                        tr.appendChild(volumeCell);

                        const changeCell = document.createElement('td');
                        const change = row.close - row.open;
                        const changePercent = (change / row.open) * 100;
                        const changeText = change.toFixed(2);
                        const percentText = changePercent.toFixed(2);
                        changeCell.textContent = `${changeText} (${percentText}%)`;
                        changeCell.className = change >= 0 ? 'up' : 'down';
                        tr.appendChild(changeCell);

                        tableBody.appendChild(tr);
                    });
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    document.getElementById('last-updated').textContent =
                        `Error: ${error.message}`;
                });
        }

        // Auto-refresh every 60 seconds
        setInterval(updateData, 60000);

        // Initial load
        updateData();
    </script>
</body>
</html>
"""


class StockDataProvider:
    """Provider for stock data."""

    def __init__(self) -> None:
        """Initialize the data provider."""
        self.data: Dict[str, pd.DataFrame] = {}

    def get_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Get stock data for a ticker."""
        if ticker not in self.data:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(period="1mo")
            self.data[ticker] = df

        df = self.data[ticker]
        if df.empty:
            return []

        result = []
        for idx, row in df.iterrows():
            if isinstance(idx, (datetime, date)):
                date_str = idx.strftime("%Y-%m-%d")
            else:
                date_str = str(idx)
            result.append(
                {
                    "date": date_str,
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                }
            )

        return result


data_provider = StockDataProvider()


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def _set_headers(self, content_type: str = "text/html") -> None:
        """Set response headers."""
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/":
            self._set_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path.startswith("/api/data"):
            self._set_headers("application/json")
            # Parse ticker from query string
            ticker = "AAPL"  # Default
            if "?" in self.path:
                query = self.path.split("?")[1]
                params = query.split("&")
                for param in params:
                    if param.startswith("ticker="):
                        ticker = param.split("=")[1]

            try:
                data = data_provider.get_data(ticker)
                self.wfile.write(json.dumps(data).encode())
            except Exception as e:
                logger.error("Error getting data: %s", e)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _generate_trade_row(self, trade: Dict[str, Any]) -> str:
        """Generate HTML for a single trade row."""
        time = trade["time"].strftime("%Y-%m-%d %H:%M:%S")
        action = "Buy" if trade["action"] == "buy" else "Sell"
        price = f"${trade['price']:.2f}"
        shares = str(trade["shares"])
        value = f"${trade['price'] * trade['shares']:.2f}"

        return (
            f"<tr><td>{time}</td>"
            f"<td>{action}</td>"
            f"<td>{price}</td>"
            f"<td>{shares}</td>"
            f"<td>{value}</td></tr>"
        )


def run_server(host: str = "localhost", port: int = 8050) -> None:
    """Run the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    logger.info("Starting fallback dashboard on http://%s:%s", host, port)
    httpd.serve_forever()


def open_browser(host: str, port: int) -> None:
    """Open the browser after a short delay."""
    time.sleep(1)
    webbrowser.open(f"http://{host}:{port}/")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Stock Dashboard")
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard on (default: 8050)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the dashboard on (default: localhost)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open the browser"
    )

    args = parser.parse_args()

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.host, args.port), daemon=True).start()

    try:
        run_server(args.host, args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Error running server: %s", e)


if __name__ == "__main__":
    main()
