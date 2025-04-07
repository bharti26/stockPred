"""Stock data handling and preprocessing utilities.

This module provides functionality for fetching, processing, and managing stock market data.
It includes methods for downloading historical data, calculating technical indicators,
and preprocessing data for machine learning models.
"""

from typing import Optional, cast, Dict, List
import pandas as pd
import yfinance as yf
import ta  # type: ignore  # No type stubs available for ta library
import numpy as np


class StockData:
    """Stock data fetching and preprocessing class.

    This class handles downloading historical stock data from Yahoo Finance,
    adding technical indicators, and preprocessing the data for use in the
    trading environment.

    Attributes:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        data (pd.DataFrame): Stock price data with technical indicators
    """

    def __init__(self, ticker: str, start_date: str, end_date: str):
        """Initialize StockData with a specific stock ticker and date range.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
            start_date (str): Start date in 'YYYY-MM-DD' forma
            end_date (str): End date in 'YYYY-MM-DD' forma
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data: Optional[pd.DataFrame] = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance.

        Downloads OHLCV (Open, High, Low, Close, Volume) data for the
        specified ticker and date range.

        Returns:
            pd.DataFrame: Historical stock data

        Raises:
            ValueError: If no data is available for the ticker
            Exception: If there's an error fetching data from Yahoo Finance
        """
        stock = yf.Ticker(self.ticker)
        # Cast the return type to ensure mypy understands it's a DataFrame
        data = cast(pd.DataFrame, stock.history(start=self.start_date, end=self.end_date))

        if data.empty:
            raise ValueError(f"No data available for {self.ticker}")

        self.data = data
        return data

    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators to the stock data.

        Calculates and adds the following technical indicators:
        - Simple Moving Averages (20, 50 days)
        - Relative Strength Index (RSI)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands

        Returns:
            pd.DataFrame: Data with added technical indicators

        Raises:
            ValueError: If no data is available (fetch_data not called)
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")

        data = self.data.copy()

        # Add Moving Averages
        data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
        data["SMA_50"] = ta.trend.sma_indicator(data["Close"], window=50)

        # Add RSI
        data["RSI"] = ta.momentum.rsi(data["Close"], window=14)

        # Add MACD
        macd = ta.trend.MACD(data["Close"])
        data["MACD"] = macd.macd()
        data["MACD_Signal"] = macd.macd_signal()

        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data["Close"])
        data["BB_Upper"] = bollinger.bollinger_hband()
        data["BB_Lower"] = bollinger.bollinger_lband()

        self.data = data
        return data

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data by handling missing values and normalizing features.

        This method:
        1. Fills missing values using forward and backward fill
        2. Normalizes all features to have zero mean and unit variance

        Returns:
            pd.DataFrame: Preprocessed data with normalized features

        Raises:
            ValueError: If no data is available (fetch_data not called)
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")

        data = self.data.copy()

        # Fill missing values
        data = data.ffill()  # Forward fill
        data = data.bfill()  # Backward fill

        # Normalize features
        features_to_normalize = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
        ]

        for feature in features_to_normalize:
            if feature in data.columns:
                mean = data[feature].mean()
                std = data[feature].std()
                data[f"{feature}_norm"] = (data[feature] - mean) / std

        self.data = data
        return data
