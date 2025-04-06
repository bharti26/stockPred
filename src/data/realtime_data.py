import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator  # type: ignore

# Add type ignore comments for modules with missing stubs
from ta.trend import MACD, SMAIndicator  # type: ignore
from ta.volatility import BollingerBands  # type: ignore


class RealTimeStockData:
    """A class to handle real-time stock data streaming and live technical analysis.

    Attributes:
        ticker (str): Stock ticker symbol
        interval (str): Data interval (1m, 5m, 15m, etc.)
        buffer_size (int): Number of historical data points to maintain
        callbacks (list): List of callback functions for data updates
    """

    def __init__(self, ticker: str, interval: str = "1m", buffer_size: int = 100):
        """Initialize real-time stock data streamer.

        Args:
            ticker: Stock ticker symbol
            interval: Data interval (1m, 5m, 15m, etc.)
            buffer_size: Number of historical data points to maintain
        """
        self.ticker = ticker.upper()  # Normalize ticker symbol
        self.interval = interval
        self.buffer_size = buffer_size
        self.data = pd.DataFrame()
        self.callbacks: List[Callable[[pd.DataFrame], None]] = []
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

        try:
            # Initialize stock info and validate ticker
            self.stock = yf.Ticker(ticker)
            info = self.stock.info
            if not info:
                raise ValueError(f"Could not get information for ticker {ticker}")
        except Exception as e:
            self.last_error = f"Error initializing ticker {ticker}: {str(e)}"
            raise ValueError(f"Invalid ticker symbol {ticker}: {str(e)}")

    def add_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add a callback function to be called on data updates.

        Args:
            callback: Function to call when data is updated
        """
        self.callbacks.append(callback)

    def _fetch_initial_data(self) -> pd.DataFrame:
        """Fetch initial historical data.

        Returns:
            DataFrame containing historical price data

        Raises:
            ValueError: If no data could be fetched
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=2)  # Get 2 days of historical data

            df = self.stock.history(start=start, end=end, interval=self.interval)

            if df.empty:
                raise ValueError(f"No historical data available for {self.ticker}")

            # Ensure we return an explicitly typed DataFrame
            result: pd.DataFrame = df.tail(self.buffer_size).copy()
            return result
        except Exception as e:
            self.last_error = f"Error fetching data: {str(e)}"
            raise ValueError(f"Failed to fetch historical data for {self.ticker}: {str(e)}")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data.

        Args:
            df: DataFrame containing price data

        Returns:
            DataFrame with added technical indicators

        Raises:
            ValueError: If indicators cannot be calculated
        """
        if len(df) == 0:
            return df.copy()

        try:
            # Create a copy to avoid modifying the original
            result = df.copy()

            # Calculate SMAs
            sma20 = SMAIndicator(close=result["Close"], window=20)
            sma50 = SMAIndicator(close=result["Close"], window=50)
            result["SMA_20"] = sma20.sma_indicator()
            result["SMA_50"] = sma50.sma_indicator()

            # Calculate Bollinger Bands
            bb = BollingerBands(close=result["Close"], window=20)
            result["BB_Upper"] = bb.bollinger_hband()
            result["BB_Lower"] = bb.bollinger_lband()

            # Calculate RSI
            rsi = RSIIndicator(close=result["Close"])
            result["RSI"] = rsi.rsi()

            # Calculate MACD
            macd = MACD(close=result["Close"])
            result["MACD"] = macd.macd()
            result["MACD_Signal"] = macd.macd_signal()

            return result
        except Exception as e:
            self.last_error = f"Error calculating indicators: {str(e)}"
            raise ValueError(f"Failed to calculate indicators: {str(e)}")

    def start_streaming(self) -> None:
        """Start the real-time data streaming."""
        if self.is_streaming:
            print("Already streaming!")
            return

        try:
            # Fetch initial data
            self.data = self._fetch_initial_data()
            if self.data.empty:
                raise ValueError(f"No initial data available for {self.ticker}")

            self.is_streaming = True

            def _stream() -> None:
                while self.is_streaming:
                    try:
                        # Fetch latest data
                        latest = self.stock.history(period="1d", interval=self.interval)

                        if latest.empty:
                            print(f"No new data available for {self.ticker}")
                            time.sleep(5)
                            continue

                        latest = latest.tail(1)

                        if not latest.empty and (
                            self.data.empty or latest.index[-1] > self.data.index[-1]
                        ):
                            # Add new data
                            self.data = pd.concat([self.data, latest]).tail(self.buffer_size)

                            try:
                                # Calculate indicators
                                self.data = self._calculate_indicators(self.data)

                                # Notify callbacks
                                for callback in self.callbacks:
                                    callback(self.data)
                            except Exception as e:
                                print(f"Error processing data: {e}")

                        # Wait for next interval
                        interval_seconds = {
                            "1m": 60,
                            "5m": 300,
                            "15m": 900,
                            "30m": 1800,
                            "1h": 3600,
                        }.get(self.interval, 60)

                        time.sleep(interval_seconds)

                    except Exception as e:
                        self.last_error = f"Error in streaming: {str(e)}"
                        print(f"Error in streaming: {e}")
                        time.sleep(5)  # Wait before retrying

            # Create and start streaming thread
            thread = threading.Thread(target=_stream)
            thread.daemon = True
            thread.start()
            self.stream_thread = thread

            print(f"Started streaming {self.ticker} data at {self.interval} intervals")
        except Exception as e:
            self.last_error = f"Error starting stream: {str(e)}"
            self.is_streaming = False
            raise ValueError(f"Failed to start streaming: {str(e)}")

    def stop_streaming(self) -> None:
        """Stop the real-time data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)  # Add timeout to avoid blocking
        print("Stopped streaming")

    def get_latest_data(self) -> pd.DataFrame:
        """Get the latest data with indicators.

        Returns:
            DataFrame containing the latest data with indicators

        Raises:
            ValueError: If no data is available
        """
        if self.data.empty:
            if self.last_error:
                raise ValueError(f"No data available: {self.last_error}")
            raise ValueError(f"No data available for {self.ticker}")
        return self.data.copy()

    def get_latest_price(self) -> Optional[float]:
        """Get the latest price.

        Returns:
            Latest closing price or None if no data is available
        """
        if not self.data.empty:
            return float(self.data["Close"].iloc[-1])
        return None

    def get_latest_indicators(self) -> Dict[str, Union[float, None]]:
        """Get the latest technical indicators.

        Returns:
            Dictionary containing latest indicator values
        """
        # Return empty dict if no data is available
        if self.data.empty:
            return {}

        # Get the latest row
        latest = self.data.iloc[-1]

        # Initialize with price and set all indicators to None initially
        result: Dict[str, Union[float, None]] = {
            "price": float(latest["Close"]),
            "sma_20": None,
            "sma_50": None,
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "bb_upper": None,
            "bb_lower": None,
        }

        # Use try-except blocks to safely access indicators
        try:
            if "SMA_20" in self.data.columns:
                result["sma_20"] = None if pd.isna(latest["SMA_20"]) else float(latest["SMA_20"])
        except (KeyError, TypeError):
            pass

        try:
            if "SMA_50" in self.data.columns:
                result["sma_50"] = None if pd.isna(latest["SMA_50"]) else float(latest["SMA_50"])
        except (KeyError, TypeError):
            pass

        try:
            if "RSI" in self.data.columns:
                result["rsi"] = None if pd.isna(latest["RSI"]) else float(latest["RSI"])
        except (KeyError, TypeError):
            pass

        try:
            if "MACD" in self.data.columns:
                result["macd"] = None if pd.isna(latest["MACD"]) else float(latest["MACD"])
        except (KeyError, TypeError):
            pass

        try:
            if "MACD_Signal" in self.data.columns:
                val = latest["MACD_Signal"]
                result["macd_signal"] = None if pd.isna(val) else float(val)
        except (KeyError, TypeError):
            pass

        try:
            if "BB_Upper" in self.data.columns:
                val = latest["BB_Upper"]
                result["bb_upper"] = None if pd.isna(val) else float(val)
        except (KeyError, TypeError):
            pass

        try:
            if "BB_Lower" in self.data.columns:
                val = latest["BB_Lower"]
                result["bb_lower"] = None if pd.isna(val) else float(val)
        except (KeyError, TypeError):
            pass

        return result
