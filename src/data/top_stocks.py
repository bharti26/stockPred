import yfinance as yf


def get_top_stocks(n: int = 10) -> list:
    """Get top n stocks by market cap from S&P 500."""
    # List of top stocks to track
    stocks = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "JPM", "V", "UNH"
    ]

    top_stocks = []
    for symbol in stocks[:n]:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            top_stocks.append({
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "price": info.get("regularMarketPrice", 0),
                "change": info.get("regularMarketChange", 0),
                "change_percent": info.get("regularMarketChangePercent", 0),
                "volume": info.get("regularMarketVolume", 0)
            })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return top_stocks
