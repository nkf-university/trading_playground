import time
import ccxt
import pandas as pd


def fetch_ohlcv(symbol: str = "SOL/USDT", timeframe: str = "5m", days: int = 30) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance public API (no keys required).
    Paginates automatically — 30 days of 5m candles = ~8,640 bars.
    Note: Binance uses USDT pairs (SOL/USDT), not USD.
    """
    exchange = ccxt.binance()

    # start timestamp in milliseconds
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    now = exchange.milliseconds()

    all_candles = []
    while since < now:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        # advance past the last fetched candle
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(0.1)  # stay well under Binance rate limit

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()
