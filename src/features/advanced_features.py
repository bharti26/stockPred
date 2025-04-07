import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore
import nltk  # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
from textblob import TextBlob  # type: ignore
from typing import List, Union, Any, cast, TypeVar, Callable
from pandas import Series, DataFrame
from numpy import float64

NumericType = Union[float64, float, int]
SeriesType = Series[Union[float64, bool]]
T = TypeVar('T', bound=Union[float64, SeriesType])

class AdvancedFeatureEngineer:
    """Class for generating advanced trading features."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df.copy()
        # Initialize sentiment analyzer
        nltk.download("vader_lexicon")
        self.sia = SentimentIntensityAnalyzer()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        # Copy the dataframe
        result = df.copy()

        # Price momentum indicators
        result["ROC"] = self._rate_of_change(result["Close"])
        result["Momentum"] = self._momentum(result["Close"])
        result["RSI"] = self._rsi(result["Close"])

        # Volatility indicators
        result["ATR"] = self._average_true_range(result)
        result["Bollinger_Bands"] = self._bollinger_bands(result["Close"])

        # Volume indicators
        result["OBV"] = self._on_balance_volume(result)
        result["Volume_MA"] = result["Volume"].rolling(window=20).mean()

        # Trend indicators
        result["MACD"] = self._macd(result["Close"])
        result["ADX"] = self._average_directional_index(result)

        # Add new advanced indicators
        result["Keltner_Channels"] = self._keltner_channels(result)
        result["Ichimoku_Cloud"] = self._ichimoku_cloud(result)
        result["Parabolic_SAR"] = self._parabolic_sar(result)
        result["Stochastic_Oscillator"] = self._stochastic_oscillator(result)
        result["Williams_%R"] = self._williams_percent_r(result)

        return result

    def add_market_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add market-wide indicators."""
        result = df.copy()

        # Get S&P 500 data for market correlation
        sp500 = yf.download("^GSPC", start=df.index[0], end=df.index[-1])
        result["SP500_Returns"] = sp500["Close"].pct_change()

        # Calculate beta
        result["Beta"] = self._calculate_beta(result["Close"].pct_change(), result["SP500_Returns"])

        # Add VIX (volatility index)
        vix = yf.download("^VIX", start=df.index[0], end=df.index[-1])
        result["VIX"] = vix["Close"]

        return result

    def add_sentiment_features(self, ticker: str) -> pd.DataFrame:
        """Add sentiment analysis features from news and social media."""
        # Get news headlines
        headlines = self._fetch_news_headlines(ticker)

        # Calculate sentiment scores
        sentiment_scores = []
        for headline in headlines:
            # VADER sentimen
            vader_score = self.sia.polarity_scores(headline)["compound"]

            # TextBlob sentimen
            blob = TextBlob(headline)
            textblob_score = blob.sentiment.polarity

            sentiment_scores.append(
                {"vader_score": vader_score, "textblob_score": textblob_score, "headline": headline}
            )

        # Create sentiment dataframe
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df["date"] = pd.Timestamp.now()

        return sentiment_df

    def add_market_microstructure(self) -> pd.DataFrame:
        """Add market microstructure features."""
        result = self.df.copy()

        # Price impact features
        result["Price_Impact"] = (result["High"] - result["Low"]) / result["Close"]
        result["Order_Flow_Imbalance"] = (result["Close"] - result["Open"]) / (
            result["High"] - result["Low"]
        )

        # Liquidity features
        result["Bid_Ask_Spread"] = self._calculate_bid_ask_spread(result)
        result["Market_Depth"] = self._calculate_market_depth(result)

        # Volume profile features
        result["Volume_Profile"] = self._calculate_volume_profile(result)
        result["Volume_Clustering"] = self._calculate_volume_clustering(result)

        return result

    def add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        result = df.copy()

        # Candlestick patterns
        result["Doji"] = self._detect_doji(result)
        result["Engulfing"] = self._detect_engulfing(result)
        result["Hammer"] = self._detect_hammer(result)
        result["Shooting_Star"] = self._detect_shooting_star(result)

        # Chart patterns
        result["Head_Shoulders"] = self._detect_head_shoulders(result)
        result["Double_Top_Bottom"] = self._detect_double_top_bottom(result)
        result["Triangle"] = self._detect_triangle(result)

        return result

    def add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        result = df.copy()

        # Volatility regime
        result["Volatility_Regime"] = self._identify_volatility_regime(result)
        result["Trend_Regime"] = self._identify_trend_regime(result)

        # Market state features
        result["Market_State"] = self._identify_market_state(result)
        result["Regime_Transition"] = self._detect_regime_transition(result)

        return result

    def _rate_of_change(self, prices: Series, period: int = 10) -> Series:
        """Calculate Rate of Change indicator."""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100

    def _momentum(self, prices: Series, period: int = 10) -> Series:
        """Calculate Momentum indicator."""
        return prices - prices.shift(period)

    def _rsi(self, prices: Series, period: int = 14) -> Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _average_true_range(self, df: DataFrame, period: int = 14) -> Series:
        """Calculate Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _bollinger_bands(
        self, prices: Series, period: int = 20, num_std: float = 2.0
    ) -> Series:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return (prices - lower_band) / (upper_band - lower_band)

    def _on_balance_volume(self, df: DataFrame) -> Series:
        """Calculate On-Balance Volume."""
        close_diff = df["Close"].diff()
        volume = df["Volume"]
        obv = (np.sign(close_diff) * volume).fillna(0).cumsum()
        return obv

    def _macd(
        self, prices: Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Series:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def _average_directional_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df["High"]
        low = df["Low"]

        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - df["Close"].shift(1))
        tr3 = abs(low - df["Close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DI and -DI
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / tr.rolling(period).mean())

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_beta(self, stock_returns: SeriesType, market_returns: SeriesType) -> float64:
        """Calculate beta coefficient."""
        # Handle potential NaN values
        valid_mask = ~(stock_returns.isna() | market_returns.isna())
        stock_returns_clean = stock_returns[valid_mask]
        market_returns_clean = market_returns[valid_mask]

        if len(stock_returns_clean) == 0 or len(market_returns_clean) == 0:
            return float64(0.0)

        covariance = stock_returns_clean.cov(market_returns_clean)
        market_variance = market_returns_clean.var()
        
        if market_variance == 0:
            return float64(0.0)
            
        return float64(covariance / market_variance)

    def _fetch_news_headlines(self, ticker: str) -> List[str]:
        """Fetch recent news headlines for a ticker."""
        # This is a placeholder - in practice, you would use a news API
        # or web scraping to get real headlines
        return [
            f"{ticker} stock shows strong performance",
            f"Analysts bullish on {ticker} future prospects",
            f"{ticker} announces new product launch",
            f"Market reacts to {ticker} earnings report",
            f"{ticker} faces regulatory challenges",
        ]

    # New helper methods for advanced features
    def _keltner_channels(
        self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0
    ) -> pd.Series:
        """Calculate Keltner Channels."""
        atr = self._average_true_range(df, period)
        middle = df["Close"].rolling(window=period).mean()
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        return (df["Close"] - lower) / (upper - lower)

    def _ichimoku_cloud(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku Cloud components."""
        high = df["High"]
        low = df["Low"]

        # Tenkan-sen (Conversion Line)
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line)
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        # Calculate cloud position
        return (df["Close"] - senkou_span_a) / (senkou_span_b - senkou_span_a)

    def _parabolic_sar(
        self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
    ) -> pd.Series:
        """Calculate Parabolic SAR."""
        high = df["High"]
        low = df["Low"]

        # Initialize variables
        trend = 1  # 1 for uptrend, -1 for downtrend
        sar = low[0]
        ep = high[0]
        af = acceleration

        # Calculate SAR
        sar_values = []
        for i in range(len(df)):
            if trend == 1:
                sar = sar + af * (ep - sar)
                if low[i] < sar:
                    trend = -1
                    sar = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:
                sar = sar + af * (ep - sar)
                if high[i] > sar:
                    trend = 1
                    sar = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
            sar_values.append(sar)

        return pd.Series(sar_values, index=df.index)

    def _stochastic_oscillator(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.Series:
        """Calculate Stochastic Oscillator."""
        high = df["High"]
        low = df["Low"]

        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((df["Close"] - lowest_low) / (highest_high - lowest_low))

        # Calculate %D
        d = k.rolling(window=d_period).mean()

        return k - d

    def _williams_percent_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high = df["High"]
        low = df["Low"]

        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        return -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)

    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact of trades."""
        return (df["Close"] - df["Open"]) / df["Volume"]

    def _calculate_order_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order imbalance."""
        return (df["Close"] - df["Open"]) / (df["High"] - df["Low"])

    def _calculate_bid_ask_spread(self, df: DataFrame) -> Series:
        """Calculate bid-ask spread."""
        # Simplified calculation using high and low prices
        spread = (df["High"] - df["Low"]) / df["Close"]
        return spread.astype(float64)

    def _calculate_market_depth(self, df: DataFrame) -> SeriesType:
        """Calculate market depth."""
        # Simplified calculation using volume and price range
        high = df["High"].astype(float64)
        low = df["Low"].astype(float64)
        volume = df["Volume"].astype(float64)
        
        price_range = self._safe_operation(lambda x, y: x - y, high, low)
        # Handle zero price range
        price_range = cast(SeriesType, price_range.replace(0, np.finfo(float64).eps))
        depth = cast(SeriesType, volume / price_range)
        return depth

    def _calculate_volume_profile(self, df: DataFrame) -> SeriesType:
        """Calculate volume profile."""
        # Simplified volume profile calculation
        volume = df["Volume"].astype(float64)
        close = df["Close"].astype(float64)
        profile = cast(SeriesType, volume * close)
        return profile

    def _calculate_volume_clustering(self, df: DataFrame) -> SeriesType:
        """Calculate volume clustering."""
        # Simplified volume clustering calculation
        volume = df["Volume"].astype(float64)
        rolling_std = volume.rolling(window=5).std()
        rolling_mean = volume.rolling(window=5).mean()
        # Handle zero mean
        rolling_mean = cast(SeriesType, rolling_mean.replace(0, np.finfo(float64).eps))
        clustering = cast(SeriesType, rolling_std / rolling_mean)
        return clustering.fillna(0)

    def _detect_doji(self, df: DataFrame) -> SeriesType:
        """Detect doji candlestick patterns."""
        close = df["Close"].astype(float64)
        open_ = df["Open"].astype(float64)
        high = df["High"].astype(float64)
        low = df["Low"].astype(float64)
        
        body = abs(close - open_)
        upper_shadow = high - df[["Open", "Close"]].astype(float64).max(axis=1)
        lower_shadow = df[["Open", "Close"]].astype(float64).min(axis=1) - low
        
        # A doji has a very small body compared to its shadows
        total_shadow = upper_shadow + lower_shadow
        # Handle zero total shadow
        total_shadow = cast(SeriesType, total_shadow.replace(0, np.finfo(float64).eps))
        is_doji = cast(SeriesType, body <= (0.1 * total_shadow))
        return is_doji.astype(float64)

    def _detect_engulfing(self, df: DataFrame) -> SeriesType:
        """Detect engulfing candlestick patterns."""
        # Convert to float64 to ensure proper comparison
        close = df["Close"].astype(float64)
        open_ = df["Open"].astype(float64)
        prev_close = close.shift(1)
        prev_open = open_.shift(1)
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(close - open_)
        
        bullish = (
            self._safe_operation(lambda x, y: x < y, open_, close)
            & self._safe_operation(lambda x, y: x < y, prev_close, prev_open)
        )
        bearish = (
            self._safe_operation(lambda x, y: x > y, open_, close)
            & self._safe_operation(lambda x, y: x > y, prev_close, prev_open)
        )
        
        engulfing = cast(
            SeriesType,
            self._safe_operation(lambda x, y: x > y, curr_body, prev_body)
            & (bullish | bearish)
        )
        return engulfing.astype(float64)

    def _detect_hammer(self, df: DataFrame) -> SeriesType:
        """Detect hammer candlestick patterns."""
        # Convert to float64 to ensure proper comparison
        close = df["Close"].astype(float64)
        open_ = df["Open"].astype(float64)
        low = df["Low"].astype(float64)
        
        body = abs(close - open_)
        lower_shadow = df[["Open", "Close"]].astype(float64).min(axis=1) - low
        
        # Handle zero body
        body = cast(SeriesType, body.replace(0, np.finfo(float64).eps))
        is_hammer = cast(
            SeriesType,
            (lower_shadow >= (2 * body)) & (body > float64(0))
        )
        return is_hammer.astype(float64)

    def _detect_shooting_star(self, df: DataFrame) -> SeriesType:
        """Detect shooting star candlestick patterns."""
        # Convert to float64 to ensure proper comparison
        close = df["Close"].astype(float64)
        open_ = df["Open"].astype(float64)
        high = df["High"].astype(float64)
        
        body = abs(close - open_)
        upper_shadow = high - df[["Open", "Close"]].astype(float64).max(axis=1)
        
        # Handle zero body
        body = cast(SeriesType, body.replace(0, np.finfo(float64).eps))
        is_shooting_star = cast(
            SeriesType,
            (upper_shadow >= (2 * body)) & (body > float64(0))
        )
        return is_shooting_star.astype(float64)

    def _detect_head_shoulders(self, df: DataFrame) -> SeriesType:
        """Detect head and shoulders pattern."""
        # Convert to float64 and ensure proper comparison
        high = df["High"].astype(float64)
        prev_high = high.shift(1)
        next_high = high.shift(-1)
        
        peaks = cast(
            SeriesType,
            (high > prev_high) & (high > next_high)
        )
        return peaks.astype(float64)

    def _detect_double_top_bottom(self, df: DataFrame) -> SeriesType:
        """Detect double top/bottom patterns."""
        # Convert to float64 and ensure proper comparison
        high = df["High"].astype(float64)
        low = df["Low"].astype(float64)
        
        prev_high = high.shift(1)
        next_high = high.shift(-1)
        prev_low = low.shift(1)
        next_low = low.shift(-1)
        
        tops = cast(
            SeriesType,
            (high > prev_high) & (high > next_high)
        )
        bottoms = cast(
            SeriesType,
            (low < prev_low) & (low < next_low)
        )
        return (tops | bottoms).astype(float64)

    def _detect_triangle(self, df: DataFrame) -> SeriesType:
        """Detect triangle patterns."""
        # Convert to float64 and ensure proper comparison
        high = df["High"].astype(float64)
        low = df["Low"].astype(float64)
        
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        
        higher_lows = cast(SeriesType, low > prev_low)
        lower_highs = cast(SeriesType, high < prev_high)
        return (higher_lows & lower_highs).astype(float64)

    def _identify_volatility_regime(self, df: DataFrame) -> SeriesType:
        """Identify volatility regime."""
        close = df["Close"].astype(float64)
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std()
        vol_mean = volatility.rolling(window=100).mean()
        high_vol = cast(SeriesType, volatility > vol_mean)
        return high_vol.astype(float64)

    def _identify_trend_regime(self, df: DataFrame) -> SeriesType:
        """Identify trend regime."""
        close = df["Close"].astype(float64)
        sma_short = close.rolling(window=20).mean()
        sma_long = close.rolling(window=50).mean()
        uptrend = cast(SeriesType, sma_short > sma_long)
        return uptrend.astype(float64)

    def _identify_market_state(self, df: DataFrame) -> SeriesType:
        """Identify market state."""
        # Combine volatility and trend regimes
        volatility = self._identify_volatility_regime(df)
        trend = self._identify_trend_regime(df)
        state = cast(
            SeriesType, (volatility.astype(float64) + trend.astype(float64)) / float64(2.0)
        )
        return state

    def _detect_regime_transition(self, df: DataFrame) -> SeriesType:
        """Detect regime transitions."""
        market_state = self._identify_market_state(df)
        transition = cast(SeriesType, market_state != market_state.shift(1))
        return transition.astype(float64)

    def add_market_microstructure_features(self) -> None:
        """Add market microstructure features."""
        # Calculate price impac
        self.df["Price_Impact"] = (self.df["High"] - self.df["Low"]) / self.df["Close"]

        # Calculate order flow imbalance
        self.df["Order_Flow_Imbalance"] = (self.df["Close"] - self.df["Open"]) / (
            self.df["High"] - self.df["Low"]
        )

    def add_volatility_features(self) -> None:
        """Add volatility-based features."""
        # Calculate rolling volatility
        self.df["Volatility"] = self.df["Close"].pct_change().rolling(window=20).std()

        # Calculate high-low range
        self.df["HL_Range"] = (self.df["High"] - self.df["Low"]) / self.df["Close"]

    def add_momentum_features(self) -> None:
        """Add momentum-based features."""
        # Calculate ROC (Rate of Change)
        self.df["ROC"] = self.df["Close"].pct_change(periods=10)

        # Calculate RSI
        delta = self.df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df["RSI"] = 100 - (100 / (1 + rs))

    def normalize_features(self) -> None:
        """Normalize all features to [0, 1] range."""
        for col in self.df.columns:
            if col not in ["Date", "Open", "High", "Low", "Close", "Volume"]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[f"{col}_norm"] = (self.df[col] - min_val) / (max_val - min_val)

    def _safe_comparison(self, a: SeriesType, b: SeriesType) -> SeriesType:
        """Safely compare two series after converting to float64."""
        return cast(SeriesType, a.astype(float64).compare(b.astype(float64)))

    def _safe_operation(self, func: Callable[[Any, Any], Any], a: T, b: T) -> T:
        """Safely perform operation after converting to float64."""
        if isinstance(a, Series) and isinstance(b, Series):
            return cast(T, func(a.astype(float64), b.astype(float64)))
        return cast(T, func(float64(a), float64(b)))
