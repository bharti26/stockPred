import threading
import time
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass

from ta.momentum import RSIIndicator  # type: ignore
from ta.trend import MACD, SMAIndicator  # type: ignore
from ta.volatility import BollingerBands  # type: ignore

from src.data.stock_data import StockData
import logging

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming settings."""
    ticker: str
    interval: str
    buffer_size: int
    stock: yf.Ticker

@dataclass
class StreamingState:
    """State-related attributes for streaming."""
    is_streaming: bool
    stream_thread: Optional[threading.Thread]
    last_error: Optional[str]
    data: pd.DataFrame
    data_buffer: Dict[str, pd.DataFrame]

class RealTimeStockData:
    """A class to handle real-time stock data streaming and live technical analysis.

    Attributes:
        ticker (str): Stock ticker symbol
        interval (str): Data interval (1d, 1h, etc.)
        buffer_size (int): Number of historical data points to maintain
        callbacks (list): List of callback functions for data updates
    """

    def __init__(self, ticker: str, interval: str = "1m", buffer_size: int = 1000) -> None:
        """Initialize real-time stock data handler.

        Args:
            ticker: Stock ticker symbol
            interval: Data interval (1d, 1h, etc.)
            buffer_size: Size of data buffer
        """
        try:
            self.ticker = ticker
            self.interval = interval
            self.buffer_size = buffer_size
            self.data = pd.DataFrame()
            self.data_buffer = {}
            self.callbacks: List[Callable[[pd.DataFrame], None]] = []
            self.is_streaming: bool = False
            self.stream_thread: Optional[threading.Thread] = None
            self.last_error: Optional[str] = None
            
            # Initialize yfinance ticker
            self.stock = yf.Ticker(ticker)
            info = self.stock.info
            if not info:
                raise ValueError(f"Could not get information for ticker {ticker}")
            
            # Fetch initial data
            self._fetch_initial_data()
            
        except Exception as e:
            error_msg = f"Error initializing ticker {ticker}: {str(e)}"
            logger.error(error_msg)
            self.last_error = error_msg
            raise ValueError(f"Invalid ticker symbol {ticker}: {str(e)}") from e

    def add_callback(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Add a callback function to be called on data updates.

        Args:
            callback: Function to call when data is updated
        """
        self.callbacks.append(callback)

    def _fetch_initial_data(self) -> pd.DataFrame:
        """Fetch initial historical data for the ticker."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # First try with daily data for 1 month
                logger.info(f"Attempt {attempt + 1}: Fetching daily data for {self.ticker}")
                data = self.stock.history(period="1mo", interval="1d")
                
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} daily data points")
                    self.data = data
                    self._update_buffer(self.data)
                    return self.data
                
                # If daily data fails, try with hourly data for 1 week
                logger.info(f"Attempt {attempt + 1}: Fetching hourly data for {self.ticker}")
                data = self.stock.history(period="1wk", interval="1h")
                
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} hourly data points")
                    self.data = data
                    self._update_buffer(self.data)
                    return self.data
                
                # If hourly data fails, try with 5-minute data for 1 day
                logger.info(f"Attempt {attempt + 1}: Fetching 5-minute data for {self.ticker}")
                data = self.stock.history(period="1d", interval="5m")
                
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} 5-minute data points")
                    self.data = data
                    self._update_buffer(self.data)
                    return self.data
                
                logger.warning(f"No data available for {self.ticker} after {attempt + 1} attempts")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error fetching data for {self.ticker} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise ValueError(f"Failed to fetch data for {self.ticker} after {max_retries} attempts: {str(e)}")
        
        return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
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

            result = pd.DataFrame(result)
            return result
        except Exception as e:
            self.last_error = f"Error calculating indicators: {str(e)}"
            raise ValueError(f"Failed to calculate indicators: {str(e)}") from e

    def start_streaming(self) -> None:
        """Start real-time data streaming."""
        if self.is_streaming:
            logger.warning("Already streaming!")
            return

        try:
            if self.data.empty:
                self._fetch_initial_data()
                if self.data.empty:
                    raise ValueError(f"No initial data available for {self.ticker}")

            self.is_streaming = True

            def _stream() -> None:
                while self.is_streaming:
                    try:
                        latest = self.stock.history(period="1d", interval=self.interval)
                        if latest.empty:
                            logger.info(f"No new data available for {self.ticker}")
                            time.sleep(5)
                            continue

                        if not latest.empty and (
                            self.data.empty or latest.index[-1] > self.data.index[-1]
                        ):
                            self.data = pd.concat([self.data, latest]).tail(self.buffer_size)
                            self.data = self._calculate_indicators(self.data)
                            self._update_buffer(self.data)
                            for callback in self.callbacks:
                                callback(self.data)
                    except Exception as e:
                        logger.error(f"Error in streaming loop: {str(e)}")
                        time.sleep(5)

            self.stream_thread = threading.Thread(target=_stream, daemon=True)
            self.stream_thread.start()

        except Exception as e:
            self.last_error = f"Error starting stream: {str(e)}"
            logger.error(f"Error starting stream: {str(e)}")
            self.is_streaming = False
            raise ValueError(f"Failed to start streaming for {self.ticker}: {str(e)}") from e

    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        logger.info("Stopped streaming")

    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Get the latest data."""
        try:
            # If we have no data, try to fetch initial data
            if self.data.empty:
                self._fetch_initial_data()
                if self.data.empty:
                    raise ValueError(f"No data available for {self.ticker}")
                return self.data.copy()

            # Try to fetch new data with the same interval as our existing data
            try:
                # First try with daily data for 1 month
                logger.info(f"Fetching daily data for {self.ticker}")
                latest = self.stock.history(period="1mo", interval="1d")
                
                if not latest.empty:
                    logger.info(f"Successfully fetched {len(latest)} daily data points")
                    self.data = latest.tail(self.buffer_size).copy()
                    self.data = self._calculate_indicators(self.data)
                    self._update_buffer(self.data)
                    return self.data.copy()
                
                # If daily data fails, try with hourly data for 1 week
                logger.info(f"Fetching hourly data for {self.ticker}")
                latest = self.stock.history(period="1wk", interval="1h")
                
                if not latest.empty:
                    logger.info(f"Successfully fetched {len(latest)} hourly data points")
                    self.data = latest.tail(self.buffer_size).copy()
                    self.data = self._calculate_indicators(self.data)
                    self._update_buffer(self.data)
                    return self.data.copy()
                
                # If hourly data fails, try with 5-minute data for 1 day
                logger.info(f"Fetching 5-minute data for {self.ticker}")
                latest = self.stock.history(period="1d", interval="5m")
                
                if not latest.empty:
                    logger.info(f"Successfully fetched {len(latest)} 5-minute data points")
                    self.data = latest.tail(self.buffer_size).copy()
                    self.data = self._calculate_indicators(self.data)
                    self._update_buffer(self.data)
                    return self.data.copy()
                
                # If we couldn't get new data but have existing data, return that
                if not self.data.empty:
                    logger.info(f"Using existing data for {self.ticker} ({len(self.data)} points)")
                    return self.data.copy()
                
                raise ValueError(f"No data available for {self.ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching new data: {str(e)}")
                if not self.data.empty:
                    logger.info(f"Using cached data for {self.ticker} due to error")
                    return self.data.copy()
                raise ValueError(f"Error getting latest data: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error getting latest data: {str(e)}")
            if not self.data.empty:
                logger.info(f"Using cached data for {self.ticker} due to error")
                return self.data.copy()
            raise ValueError(f"Error getting latest data: {str(e)}")

    def get_latest_price(self) -> Optional[float]:
        """Get the latest closing price."""
        if not self.data.empty:
            return float(self.data["Close"].iloc[-1])
        return None

    def _get_indicator_value(self, latest: pd.Series, column: str) -> Optional[float]:
        """Safely get an indicator value from the latest data."""
        try:
            if column in self.data.columns:
                val = latest[column]
                return None if pd.isna(val) else float(val)
        except Exception:
            pass
        return None

    def _initialize_indicator_result(self, latest: pd.Series) -> Dict[str, Union[float, None]]:
        """Initialize the result dictionary for indicators."""
        return {
            "sma_20": self._get_indicator_value(latest, "SMA_20"),
            "sma_50": self._get_indicator_value(latest, "SMA_50"),
            "rsi": self._get_indicator_value(latest, "RSI"),
            "macd": self._get_indicator_value(latest, "MACD"),
            "macd_signal": self._get_indicator_value(latest, "MACD_Signal"),
            "bb_upper": self._get_indicator_value(latest, "BB_Upper"),
            "bb_lower": self._get_indicator_value(latest, "BB_Lower"),
        }

    def get_latest_indicators(self) -> Dict[str, Union[float, None]]:
        """Get the latest technical indicators."""
        if self.data.empty:
            return {}

        latest = self.data.iloc[-1]
        return self._initialize_indicator_result(latest)

    def get_current_values(self) -> Dict[str, Union[float, str]]:
        """Get current stock values from yfinance.
        
        Returns:
            Dictionary containing current price, change, volume, and market cap
        """
        try:
            logger.info(f"Fetching current values for {self.ticker}")
            
            # If we don't have data yet, fetch initial data
            if self.data.empty:
                logger.info("No data available, fetching initial data")
                self._fetch_initial_data()
            
            # Get the latest data point
            if not self.data.empty:
                latest = self.data.iloc[-1]
                logger.info(f"Latest data point: {latest}")
                
                return {
                    "current_price": latest["Close"],
                    "change": ((latest["Close"] - latest["Open"]) / latest["Open"]) * 100,
                    "change_percent": ((latest["Close"] - latest["Open"]) / latest["Open"]) * 100,
                    "volume": latest["Volume"],
                    "market_cap": self.stock.info.get("marketCap", 0)
                }
            
            # If we still don't have data, try to get it from yfinance directly
            info = self.stock.info
            logger.info(f"Raw info data: {info}")
            
            current_price = info.get("regularMarketPrice", 0.0)
            change = info.get("regularMarketChange", 0.0)
            change_percent = info.get("regularMarketChangePercent", 0.0)
            volume = info.get("regularMarketVolume", 0)
            market_cap = info.get("marketCap", 0)
            
            logger.info(f"Extracted values - Price: {current_price}, Change: {change}, Volume: {volume}")
            
            return {
                "current_price": current_price,
                "change": change,
                "change_percent": change_percent,
                "volume": volume,
                "market_cap": market_cap
            }
            
        except Exception as e:
            logger.error(f"Error getting current values: {str(e)}")
            return {
                "current_price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "market_cap": 0
            }

    def fetch_realtime_data(self) -> Optional[pd.DataFrame]:
        """Fetch real-time data."""
        try:
            latest = self.stock.history(period="1d", interval=self.interval)
            if not latest.empty:
                return latest
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
        return None

    def _update_buffer(self, data: pd.DataFrame) -> None:
        """Update the data buffer with new data."""
        if self.ticker not in self.data_buffer:
            self.data_buffer[self.ticker] = data
            return
            
        self.data_buffer[self.ticker] = pd.concat(
            [self.data_buffer[self.ticker], data]
        ).drop_duplicates().tail(self.buffer_size)

    def process_realtime_data(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Process real-time data."""
        try:
            self.data_buffer[ticker] = pd.concat(
                [self.data_buffer.get(ticker, pd.DataFrame()), df]
            ).tail(self.buffer_size)

            processed_df = self.data_buffer[ticker].copy()
            processed_df = self.add_technical_indicators(processed_df)
            processed_df = self.preprocess_data(processed_df)
            return processed_df
        except Exception as e:
            logger.error(f"Error processing real-time data: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        try:
            stock_data = StockData(self.ticker, "", "")
            stock_data.data = df.copy()
            return stock_data.add_technical_indicators()
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            stock_data = StockData(self.ticker, "", "")
            stock_data.data = df.copy()
            return stock_data.preprocess_data()
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
