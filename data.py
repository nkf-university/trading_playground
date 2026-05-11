import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

_client = CryptoHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


def fetch_ohlcv(symbol: str = "SOL/USD", timeframe: str = "5m", days: int = 1) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Alpaca crypto data API.
    timeframe: '1m', '5m', '15m', '1h', '1d'
    """
    unit_map = {"m": TimeFrameUnit.Minute, "h": TimeFrameUnit.Hour, "d": TimeFrameUnit.Day}
    amount = int("".join(c for c in timeframe if c.isdigit()))
    unit = unit_map[timeframe[-1]]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount, unit),
        start=start,
        end=end,
    )
    bars = _client.get_crypto_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars.sort_index()
