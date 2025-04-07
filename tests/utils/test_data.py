from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def generate_sample_price_data(
    days: int = 30,
    initial_price: float = 100.0,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Generate sample price and volume data.

    Args:
        days: Number of days of data to generate.
        initial_price: Starting price for the asset.
        start_date: Starting date for the data. If None, uses current date - days.

    Returns:
        DataFrame with columns: Close, Volume
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    dates = [start_date + timedelta(days=x) for x in range(days)]
    price = initial_price
    prices = []
    volumes = []

    for _ in range(days):
        price *= 1 + np.random.normal(0.0002, 0.02)
        volume = np.random.randint(50000, 500000)
        prices.append(price)
        volumes.append(volume)

    return pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates) 
