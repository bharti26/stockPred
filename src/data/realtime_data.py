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
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                raise ValueError(f"Could not get information for ticker {ticker}")
            
            self.config = StreamingConfig(
                ticker=ticker,
                interval=interval,
                buffer_size=buffer_size,
                stock=stock
            )
            
            self.state = StreamingState(
                is_streaming=False,
                stream_thread=None,
                last_error=None,
                data=pd.DataFrame(),
                data_buffer={}
            )
            
            self.callbacks: List[Callable[[pd.DataFrame], None]] = []
            self.stock_data = StockData(ticker, interval=interval)
            
        except Exception as e:
            error_msg = f"Error initializing ticker {ticker}: {str(e)}"
            logger.error(error_msg)
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
                intervals = ["1d", "1h", "5m", "1m"]
                periods = ["1mo", "1wk", "5d", "1d"]

                for interval in intervals:
                    for period in periods:
                        logger.info(
                            f"Attempting to fetch {self.config.ticker} data with "
                            f"interval {interval} and period {period}"
                        )
                        
                        df = self.config.stock.history(period=period, interval=interval)
                        if df.empty:
                            logger.info(
                                f"No data returned for {self.config.ticker} with "
                                f"interval {interval} and period {period}"
                            )
                            continue
                            
                        logger.info(
                            f"Successfully fetched {len(df)} data points "
                            f"for {self.config.ticker}"
                        )
                        self.config.interval = interval
                        return df.tail(self.config.buffer_size).copy()

                if attempt < max_retries - 1:
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} after "
                        f"{retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    continue
                
                raise ValueError(
                    f"No historical data available for {self.config.ticker} "
                    f"after {max_retries} attempts"
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                
                self.state.last_error = f"Error fetching data: {str(e)}"
                raise ValueError(
                    f"Failed to fetch historical data for {self.config.ticker}: "
                    f"{str(e)}"
                ) from e

        raise ValueError(
            f"No historical data available for {self.config.ticker} "
            f"after {max_retries} attempts"
        )

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
            self.state.last_error = f"Error calculating indicators: {str(e)}"
            raise ValueError(f"Failed to calculate indicators: {str(e)}") from e

    def start_streaming(self) -> None:
        """Start real-time data streaming."""
        if self.state.is_streaming:
            logger.warning("Already streaming!")
            return

        try:
            # Fetch initial data
            try:
                self.state.data = self._fetch_initial_data()
                if self.state.data.empty:
                    raise ValueError(f"No initial data available for {self.config.ticker}")
            except Exception as e:
                logger.warning(f"Could not fetch initial data: {str(e)}")
                # Create empty DataFrame with required columns
                self.state.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
                logger.info("Created empty DataFrame as fallback")

            self.state.is_streaming = True

            def _stream() -> None:
                while self.state.is_streaming:
                    try:
                        latest = self.config.stock.history(period="1d", interval=self.config.interval)
                        if latest.empty:
                            logger.info(f"No new data available for {self.config.ticker}")
                            time.sleep(5)
                            continue

                        if not latest.empty and (
                            self.state.data.empty or latest.index[-1] > self.state.data.index[-1]
                        ):
                            self.state.data = pd.concat([self.state.data, latest]).tail(self.config.buffer_size)
                            self.state.data = self._calculate_indicators(self.state.data)
                            for callback in self.callbacks:
                                callback(self.state.data)
                    except Exception as e:
                        logger.error(f"Error in streaming loop: {str(e)}")
                        time.sleep(5)

            self.state.stream_thread = threading.Thread(target=_stream, daemon=True)
            self.state.stream_thread.start()

        except Exception as e:
            self.state.last_error = f"Error starting stream: {str(e)}"
            logger.error(f"Error starting stream: {str(e)}")
            self.state.is_streaming = False
            raise ValueError(f"Failed to start streaming for {self.config.ticker}: {str(e)}") from e

    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self.state.is_streaming = False
        if self.state.stream_thread:
            self.state.stream_thread.join(timeout=1.0)
        logger.info("Stopped streaming")

    def get_latest_data(self) -> pd.DataFrame:
        """Get the latest data."""
        if self.state.data.empty:
            if self.state.last_error:
                raise ValueError(f"No data available: {self.state.last_error}")
            raise ValueError(f"No data available for {self.config.ticker}")

        try:
            if self.state.is_streaming:
                latest = self.config.stock.history(period="1d", interval=self.config.interval)
                if not latest.empty:
                    latest = latest.tail(1)
                    if not self.state.data.empty and latest.index[-1] > self.state.data.index[-1]:
                        self.state.data = pd.concat([self.state.data, latest]).tail(self.config.buffer_size)
                        self.state.data = self._calculate_indicators(self.state.data)
                        logger.info(f"Updated data for {self.config.ticker} with {len(latest)} new points")
                    return self.state.data.copy()
                
                logger.info(f"No new data available for {self.config.ticker}")
                return self.state.data.copy()

            return self.state.data.copy()
        except Exception as e:
            if not self.state.data.empty:
                logger.warning(f"Using cached data for {self.config.ticker} due to error: {str(e)}")
                return self.state.data.copy()
            raise ValueError(f"Error getting latest data: {str(e)}")

    def get_latest_price(self) -> Optional[float]:
        """Get the latest closing price."""
        if not self.state.data.empty:
            return float(self.state.data["Close"].iloc[-1])
        return None

    def _get_indicator_value(self, latest: pd.Series, column: str) -> Optional[float]:
        """Safely get an indicator value from the latest data."""
        try:
            if column in self.state.data.columns:
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
        if self.state.data.empty:
            return {}

        latest = self.state.data.iloc[-1]
        return self._initialize_indicator_result(latest)

    def fetch_realtime_data(self) -> Optional[pd.DataFrame]:
        """Fetch real-time data."""
        try:
            latest = self.config.stock.history(period="1d", interval=self.config.interval)
            if not latest.empty:
                return latest
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
        return None

    def _update_buffer(self, data: pd.DataFrame) -> None:
        """Update the data buffer with new data."""
        if self.config.ticker not in self.state.data_buffer:
            self.state.data_buffer[self.config.ticker] = data
            return
            
        self.state.data_buffer[self.config.ticker] = pd.concat(
            [self.state.data_buffer[self.config.ticker], data]
        ).drop_duplicates().tail(self.config.buffer_size)

    def process_realtime_data(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Process real-time data."""
        try:
            self.state.data_buffer[ticker] = pd.concat(
                [self.state.data_buffer.get(ticker, pd.DataFrame()), df]
            ).tail(self.config.buffer_size)

            processed_df = self.state.data_buffer[ticker].copy()
            processed_df = self.add_technical_indicators(processed_df)
            processed_df = self.preprocess_data(processed_df)
            return processed_df
        except Exception as e:
            logger.error(f"Error processing real-time data: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        try:
            stock_data = StockData(self.config.ticker, "", "")
            stock_data.data = df.copy()
            return stock_data.add_technical_indicators()
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            stock_data = StockData(self.config.ticker, "", "")
            stock_data.data = df.copy()
            return stock_data.preprocess_data()
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
