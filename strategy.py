from __future__ import annotations
import pandas as pd

# ── Bollinger Band parameters ──────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2.0

# ── RSI filter thresholds ──────────────────────────────────────────────
RSI_PERIOD = 14
RSI_LONG_MAX = 45    # only go long if RSI is below this (confirms oversold)
RSI_SHORT_MIN = 55   # only go short if RSI is above this (confirms overbought)

# ── Volume filter ──────────────────────────────────────────────────────
VOL_PERIOD = 20
VOL_MULTIPLIER = 1.2  # current bar volume must be > 1.2x the rolling average


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bollinger Bands: rolling mean ± N standard deviations
    sma = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = sma + BB_STD * std
    df["bb_middle"] = sma
    df["bb_lower"] = sma - BB_STD * std

    # RSI via Wilder's smoothing (EWM with com = period - 1)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # Volume moving average for the activity filter
    df["vol_ma"] = df["volume"].rolling(VOL_PERIOD).mean()

    return df


def get_signal(row: pd.Series, prev_row: pd.Series) -> str | None:
    """
    Return 'long', 'short', or None for the current bar.

    LONG  — close crosses BELOW lower BB  AND  RSI < 45  AND  volume spike
    SHORT — close crosses ABOVE upper BB  AND  RSI > 55  AND  volume spike
    """
    # skip bar if any indicator hasn't warmed up yet
    cols = ["bb_upper", "bb_middle", "bb_lower", "rsi", "vol_ma"]
    if row[cols].isna().any() or prev_row[cols].isna().any():
        return None

    # volume filter: this bar must show above-average activity
    if row["volume"] <= VOL_MULTIPLIER * row["vol_ma"]:
        return None

    # LONG: previous close was inside/above lower band, current close broke below
    if prev_row["close"] >= prev_row["bb_lower"] and row["close"] < row["bb_lower"]:
        if row["rsi"] < RSI_LONG_MAX:
            return "long"

    # SHORT: previous close was inside/below upper band, current close broke above
    if prev_row["close"] <= prev_row["bb_upper"] and row["close"] > row["bb_upper"]:
        if row["rsi"] > RSI_SHORT_MIN:
            return "short"

    return None


def should_exit(row: pd.Series, side: str) -> bool:
    """Exit when price reverts to the middle band (20-period SMA)."""
    if side == "long":
        # price recovered back up to the mean
        return row["close"] >= row["bb_middle"]
    if side == "short":
        # price fell back down to the mean
        return row["close"] <= row["bb_middle"]
    return False
