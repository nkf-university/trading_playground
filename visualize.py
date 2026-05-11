"""
Run:  python visualize.py
Opens browser charts for every tracked symbol / pair.
  - Crypto pairs: price + RSI with signal markers
  - Stock symbols: price + SMA20/SMA50 with crossover markers
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config
from strategies.crypto_rsi import rsi as compute_rsi
from strategies.moving_average import sma_crossover_signal

# ---- what to chart ---------------------------------------------------
CRYPTO_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LINK/USD", "AAVE/USD", "DOT/USD"]
STOCK_SYMBOLS = ["AAPL"]
# ----------------------------------------------------------------------

crypto_client = CryptoHistoricalDataClient(config.API_KEY, config.SECRET_KEY)
stock_client = StockHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


# ── data fetchers ──────────────────────────────────────────────────────

def fetch_crypto(symbol: str, days: int = 7) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end)
    bars = crypto_client.get_crypto_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars.sort_index()


def fetch_stock(symbol: str, days: int = 120) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = stock_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    bars.index = bars.index.tz_convert("UTC")
    return bars.sort_index()


# ── signal history helpers ─────────────────────────────────────────────

def rsi_signals(bars: pd.DataFrame, period: int = 14, oversold=30, overbought=70):
    rsi_series = compute_rsi(bars["close"], period)
    buys = bars[rsi_series < oversold]
    sells = bars[rsi_series > overbought]
    return rsi_series, buys, sells


def sma_signals(bars: pd.DataFrame, short=20, long=50):
    sma_short = bars["close"].rolling(short).mean()
    sma_long = bars["close"].rolling(long).mean()
    prev_short = sma_short.shift(1)
    prev_long = sma_long.shift(1)
    buy_mask = (prev_short <= prev_long) & (sma_short > sma_long)
    sell_mask = (prev_short >= prev_long) & (sma_short < sma_long)
    return sma_short, sma_long, bars[buy_mask], bars[sell_mask]


# ── chart builders ─────────────────────────────────────────────────────

def chart_crypto(symbol: str):
    print(f"Fetching {symbol}...")
    bars = fetch_crypto(symbol)
    rsi_series, buys, sells = rsi_signals(bars)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=(f"{symbol} — Price (1h)", "RSI (14)"),
    )

    # price line
    fig.add_trace(go.Scatter(
        x=bars.index, y=bars["close"],
        line=dict(color="#5B8DEF", width=1.5),
        name="Price",
    ), row=1, col=1)

    # buy / sell markers on price
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["close"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=10, color="#2ECC71"),
        name="Buy signal",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["close"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="#E74C3C"),
        name="Sell signal",
    ), row=1, col=1)

    # RSI line
    fig.add_trace(go.Scatter(
        x=bars.index, y=rsi_series,
        line=dict(color="#F39C12", width=1.5),
        name="RSI",
    ), row=2, col=1)

    # oversold / overbought bands
    fig.add_hrect(y0=0, y1=30, fillcolor="#2ECC71", opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="#E74C3C", opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#2ECC71", width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#E74C3C", width=1, dash="dot"), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — RSI Mean Reversion",
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        height=600,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.show()


def chart_stock(symbol: str):
    print(f"Fetching {symbol}...")
    bars = fetch_stock(symbol)
    sma_short, sma_long, buys, sells = sma_signals(bars)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bars.index, y=bars["close"],
        line=dict(color="#5B8DEF", width=1.5),
        name="Price",
    ))
    fig.add_trace(go.Scatter(
        x=bars.index, y=sma_short,
        line=dict(color="#F39C12", width=1.2, dash="dot"),
        name="SMA 20",
    ))
    fig.add_trace(go.Scatter(
        x=bars.index, y=sma_long,
        line=dict(color="#9B59B6", width=1.2, dash="dot"),
        name="SMA 50",
    ))
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["close"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="#2ECC71"),
        name="Golden cross (buy)",
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["close"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="#E74C3C"),
        name="Death cross (sell)",
    ))

    fig.update_layout(
        title=f"{symbol} — SMA 20/50 Crossover",
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        yaxis_title="Price (USD)",
    )
    fig.show()


# ── main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for pair in CRYPTO_PAIRS:
        chart_crypto(pair)
    for sym in STOCK_SYMBOLS:
        chart_stock(sym)
