import threading
import time
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
            raise ValueError(f"Invalid ticker symbol {ticker}: {str(e)}") from e

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
            ValueError: If no data could be fetched after all retries
        """
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Try different intervals and periods to get data
                intervals = ["1d", "1h", "5m", "1m"]  # Try daily, hourly, 5-minute, then 1-minute intervals
                periods = ["1mo", "1wk", "5d", "1d"]  # Try 1 month, 1 week, 5 days, then 1 day

                for interval in intervals:
                    for period in periods:
                        try:
                            print(f"Attempting to fetch {self.ticker} data with interval {interval} and period {period}")
                            # Try to get data from yfinance
                            df = self.stock.history(period=period, interval=interval)
                            
                            if not df.empty:
                                print(f"Successfully fetched {len(df)} data points for {self.ticker}")
                                # Store the successful interval
                                self.interval = interval
                                # Ensure we return an explicitly typed DataFrame
                                result: pd.DataFrame = df.tail(self.buffer_size).copy()
                                return result
                            else:
                                print(f"No data returned for {self.ticker} with interval {interval} and period {period}")
                        except Exception as e:
                            print(f"Failed to fetch data with interval {interval} and period {period}: {str(e)}")
                            continue

                # If we get here, we couldn't fetch data with any combination
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(f"No historical data available for {self.ticker} after {max_retries} attempts")

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                else:
                    self.last_error = f"Error fetching data: {str(e)}"
                    raise ValueError(f"Failed to fetch historical data for {self.ticker}: {str(e)}") from e

        # This should never be reached due to the raise in the loop
        raise ValueError(f"No historical data available for {self.ticker} after {max_retries} attempts")

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
            raise ValueError(f"Failed to calculate indicators: {str(e)}") from e

    def start_streaming(self) -> None:
        """Start the real-time data streaming."""
        if self.is_streaming:
            print("Already streaming!")
            return

        try:
            # Fetch initial data
            try:
                self.data = self._fetch_initial_data()
                if self.data.empty:
                    raise ValueError(f"No initial data available for {self.ticker}")
            except Exception as e:
                print(f"Warning: Could not fetch initial data: {str(e)}")
                # Create empty DataFrame with required columns
                self.data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                print("Created empty DataFrame as fallback")

            self.is_streaming = True

            def _stream() -> None:
                while self.is_streaming:
                    try:
                        # Fetch latest data using the determined interval
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
                                print(f"Error calculating indicators: {str(e)}")
                                # Continue streaming even if indicators fail
                                continue

                        time.sleep(5)  # Wait before next update
                    except Exception as e:
                        print(f"Error in streaming loop: {str(e)}")
                        time.sleep(5)  # Wait before retrying
                        continue

            # Start streaming in a separate thread
            self.stream_thread = threading.Thread(target=_stream, daemon=True)
            self.stream_thread.start()

        except Exception as e:
            self.last_error = f"Error starting stream: {str(e)}"
            print(f"Error starting stream: {str(e)}")
            self.is_streaming = False
            raise ValueError(f"Failed to start streaming for {self.ticker}: {str(e)}") from e

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

        try:
            # Try to fetch latest data if streaming is active
            if self.is_streaming:
                try:
                    latest = self.stock.history(period="1d", interval=self.interval)
                    if not latest.empty:
                        latest = latest.tail(1)
                        if not self.data.empty and latest.index[-1] > self.data.index[-1]:
                            self.data = pd.concat([self.data, latest]).tail(self.buffer_size)
                            self.data = self._calculate_indicators(self.data)
                            print(f"Updated data for {self.ticker} with {len(latest)} new points")
                    else:
                        print(f"No new data available for {self.ticker}")
                        
                        # Try alternative data source if yfinance fails
                        try:
                            print(f"Trying alternative data source for {self.ticker}")
                            # Try to get data from a different source (e.g., Alpha Vantage)
                            # This is a placeholder for alternative data source
                            # You would need to implement the actual alternative data source
                            pass
                        except Exception as alt_e:
                            print(f"Alternative data source also failed: {str(alt_e)}")
                            
                except Exception as e:
                    print(f"Warning: Failed to fetch latest data: {str(e)}")
                    # Continue with existing data if available

            return self.data.copy()
        except Exception as e:
            # If we can't get latest data, return what we have
            if not self.data.empty:
                print(f"Using cached data for {self.ticker} due to error: {str(e)}")
                return self.data.copy()
            raise ValueError(f"Error getting latest data: {str(e)}")

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
