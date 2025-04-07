from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.stock_data import StockData


@pytest.fixture
def stock_data() -> StockData:
    """Create stock data instance for testing"""
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    return StockData(ticker, start_date, end_date)


def test_initialization(stock_data: StockData) -> None:
    """Test if StockData is correctly initialized"""
    assert stock_data.ticker == "AAPL"
    assert stock_data.start_date == "2023-01-01"
    assert stock_data.end_date == "2023-12-31"
    assert stock_data.data is None


@patch("yfinance.Ticker")
def test_fetch_data(mock_ticker: MagicMock, stock_data: StockData) -> None:
    """Test data fetching from Yahoo Finance"""
    # Create mock data
    mock_data = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
            "Volume": [1000, 1100, 1200],
        }
    )

    # Configure mock
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_data
    mock_ticker.return_value = mock_ticker_instance

    # Fetch data
    data = stock_data.fetch_data()

    # Verify the data
    assert data is not None
    assert len(data) == 3
    assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])


def test_add_technical_indicators(stock_data: StockData) -> None:
    """Test adding technical indicators"""
    # Create sample data
    stock_data.data = pd.DataFrame(
        {
            "Open": np.linspace(100, 200, 100),
            "High": np.linspace(105, 205, 100),
            "Low": np.linspace(95, 195, 100),
            "Close": np.linspace(102, 202, 100),
            "Volume": np.random.randint(1000, 2000, 100),
        }
    )

    # Add technical indicators
    data = stock_data.add_technical_indicators()

    # Verify indicators are added
    expected_indicators = ["SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    assert all(indicator in data.columns for indicator in expected_indicators)


def test_add_technical_indicators_no_data(stock_data: StockData) -> None:
    """Test adding technical indicators with no data"""
    with pytest.raises(ValueError):
        stock_data.add_technical_indicators()


def test_preprocess_data(stock_data: StockData) -> None:
    """Test data preprocessing"""
    # Create sample data with missing values
    stock_data.data = pd.DataFrame(
        {
            "Open": [100, np.nan, 102],
            "High": [105, 106, np.nan],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
            "Volume": [1000, np.nan, 1200],
            "SMA_20": [100, 101, 102],
            "SMA_50": [99, 100, 101],
            "RSI": [60, 65, 70],
            "MACD": [0.5, 0.6, 0.7],
            "MACD_Signal": [0.4, 0.5, 0.6],
            "BB_Upper": [110, 111, 112],
            "BB_Lower": [90, 91, 92],
        }
    )
    # Preprocess data
    processed_data = stock_data.preprocess_data()

    # Check if missing values are filled
    assert not processed_data.isnull().any().any()
    # Check if normalized features are added
    expected_norm_features = [
        "Open_norm",
        "High_norm",
        "Low_norm",
        "Close_norm",
        "Volume_norm",
        "SMA_20_norm",
        "SMA_50_norm",
        "RSI_norm",
        "MACD_norm",
        "MACD_Signal_norm",
        "BB_Upper_norm",
        "BB_Lower_norm",
    ]
    assert all(feature in processed_data.columns for feature in expected_norm_features)

    # Check if normalized features have mean close to 0 and std close to 1
    for feature in expected_norm_features:
        assert abs(processed_data[feature].mean()) < 0.1
        assert abs(processed_data[feature].std() - 1) < 0.1


def test_preprocess_data_no_data(stock_data: StockData) -> None:
    """Test preprocessing with no data"""
    with pytest.raises(ValueError):
        stock_data.preprocess_data()
