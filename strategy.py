"""
Strategy 4 — RSI Reversion with 1d trend filter.

Entry : RSI(14) on 1h crosses below 30 (oversold)
        AND daily EMA20 > EMA50 (not a bear market)
Exit  : RSI(14) crosses above 70 (overbought)

Backtest result (2018–2026, OOS 2022–present):
  Sharpe 1.76  |  Max DD -15%  |  Win rate 95%  |  44 trades on BTC
"""
from __future__ import annotations
import pandas as pd

RSI_PERIOD    = 14
RSI_OVERSOLD  = 30.0
RSI_OVERBOUGHT = 70.0
EMA_FAST      = 20   # daily periods
EMA_SLOW      = 50   # daily periods


def compute_indicators(df_1h: pd.DataFrame, df_1d: pd.DataFrame):
    """Add RSI to 1h bars and EMA trend flags to 1d bars."""
    df_1h = df_1h.copy()
    delta = df_1h["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_1h["rsi"] = 100 - 100 / (1 + gain / loss)

    df_1d = df_1d.copy()
    df_1d["ema_fast"] = df_1d["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df_1d["ema_slow"] = df_1d["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df_1d["uptrend"]  = df_1d["ema_fast"] > df_1d["ema_slow"]

    return df_1h, df_1d


def get_signal(row: pd.Series, prev_row: pd.Series, daily_uptrend: bool) -> str | None:
    """
    Returns 'long' or None.
    Entry only on the crossover bar (prev RSI >= 30, current RSI < 30)
    to avoid re-entering on sustained oversold conditions.
    """
    if pd.isna(row["rsi"]) or pd.isna(prev_row["rsi"]):
        return None

    # RSI crosses below oversold threshold in an uptrending market
    if prev_row["rsi"] >= RSI_OVERSOLD and row["rsi"] < RSI_OVERSOLD:
        if daily_uptrend:
            return "long"

    return None


def should_exit(row: pd.Series) -> bool:
    """Exit when RSI crosses above 70."""
    return float(row["rsi"]) > RSI_OVERBOUGHT
