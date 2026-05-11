import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def rsi_signal(
    bars: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[str, float]:
    """
    Returns (signal, rsi_value): signal is 'buy', 'sell', or 'hold'.
    bars must have a 'close' column, sorted oldest-first.
    """
    if len(bars) < period + 1:
        return "hold", float("nan")

    rsi_val = rsi(bars["close"], period).iloc[-1]

    if rsi_val < oversold:
        return "buy", rsi_val
    if rsi_val > overbought:
        return "sell", rsi_val
    return "hold", rsi_val
