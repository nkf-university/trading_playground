"""
Live paper trading — Strategy 4: RSI Reversion with 1d trend filter.

Entry : RSI(14) on 1h crosses below 30 AND daily EMA20 > EMA50
Exit  : RSI(14) crosses above 70
Pairs : BTC/USD and ETH/USD only (validated in backtest)

Run:   python live.py
Stop:  Ctrl+C
"""

import csv
import json
import os
import time
import signal
import sys
from datetime import datetime, timezone, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config
from data     import fetch_ohlcv
from strategy import compute_indicators, get_signal, should_exit

# ── config ─────────────────────────────────────────────────────────────
PAIRS          = ["BTC/USD", "ETH/USD"]
TIMEFRAME_1H   = "1h"
TIMEFRAME_1D   = "1d"
CANDLE_MINUTES = 60          # strategy runs on 1h candles
ORDER_SIZE_USD = 10.0
TRADES_FILE    = "trades.csv"
STATE_FILE     = ".live_state.json"
# ───────────────────────────────────────────────────────────────────────

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)


# ── Alpaca helpers ─────────────────────────────────────────────────────

def alpaca_symbol(pair: str) -> str:
    return pair.replace("/", "")


def get_qty(pair: str) -> float:
    try:
        pos = trading_client.get_open_position(alpaca_symbol(pair))
        return float(pos.qty)
    except Exception:
        return 0.0


def place_buy(pair: str):
    return trading_client.submit_order(MarketOrderRequest(
        symbol=pair, notional=ORDER_SIZE_USD,
        side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
    ))


def place_sell(pair: str, qty: float):
    return trading_client.submit_order(MarketOrderRequest(
        symbol=pair, qty=qty,
        side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
    ))


# ── state ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ── trade logging ──────────────────────────────────────────────────────

def log_trade(trade: dict):
    fieldnames = ["timestamp", "ticker", "side", "entry_price",
                  "exit_price", "pnl_pct", "hold_time_minutes"]
    write_header = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(trade)


# ── timing ─────────────────────────────────────────────────────────────

def seconds_until_next_candle(candle_minutes: int) -> float:
    now = datetime.now(timezone.utc)
    seconds_past = (now.minute % candle_minutes) * 60 + now.second
    return candle_minutes * 60 - seconds_past + 10   # +10s buffer


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── per-pair tick ──────────────────────────────────────────────────────

def process_pair(pair: str, state: dict) -> dict:
    try:
        df_1h = fetch_ohlcv(pair, TIMEFRAME_1H, days=5)    # ~120 bars — enough for RSI warmup
        df_1d = fetch_ohlcv(pair, TIMEFRAME_1D, days=120)  # 120 days — enough for EMA50
        df_1h, df_1d = compute_indicators(df_1h, df_1d)
    except Exception as e:
        print(f"  [{pair}] ERROR fetching data: {e}")
        return state

    if len(df_1h) < 3 or len(df_1d) < 2:
        return state

    # use last two CLOSED 1h candles (avoid partially-formed candle)
    row      = df_1h.iloc[-2]
    prev_row = df_1h.iloc[-3]
    candle_time = row.name.strftime("%Y-%m-%d %H:%M UTC")

    # daily trend: use last closed daily candle
    daily_uptrend = bool(df_1d.iloc[-2]["uptrend"])

    qty          = get_qty(pair)
    in_position  = qty > 0
    pair_state   = state.get(pair)

    # ── exit ──────────────────────────────────────────────────────────
    if in_position and pair_state:
        if should_exit(row):
            try:
                place_sell(pair, qty)
                entry_price  = pair_state["entry_price"]
                entry_time   = datetime.fromisoformat(pair_state["entry_time"])
                exit_price   = row["close"]
                hold_minutes = (row.name.to_pydatetime() - entry_time).total_seconds() / 60
                pnl_pct      = (exit_price - entry_price) / entry_price * 100

                log_trade({
                    "timestamp":         pair_state["entry_time"],
                    "ticker":            pair,
                    "side":              "long",
                    "entry_price":       round(entry_price, 6),
                    "exit_price":        round(exit_price, 6),
                    "pnl_pct":           round(pnl_pct, 4),
                    "hold_time_minutes": round(hold_minutes, 1),
                })
                print(f"  [{pair}] EXIT   rsi={row['rsi']:.1f}  "
                      f"entry={entry_price:.4f}  exit={exit_price:.4f}  "
                      f"pnl={pnl_pct:+.3f}%  hold={hold_minutes:.0f}m  [{candle_time}]")
                state.pop(pair, None)
            except Exception as e:
                print(f"  [{pair}] ERROR placing sell: {e}")
        else:
            unr = (row["close"] - pair_state["entry_price"]) / pair_state["entry_price"] * 100
            print(f"  [{pair}] IN POSITION  rsi={row['rsi']:.1f}  "
                  f"entry={pair_state['entry_price']:.4f}  current={row['close']:.4f}  "
                  f"unrealised={unr:+.3f}%  trend={'↑' if daily_uptrend else '↓'}  [{candle_time}]")
        return state

    # ── entry ─────────────────────────────────────────────────────────
    sig = get_signal(row, prev_row, daily_uptrend)
    if sig == "long":
        try:
            place_buy(pair)
            state[pair] = {
                "entry_price": row["close"],
                "entry_time":  row.name.isoformat(),
            }
            print(f"  [{pair}] ENTRY  rsi={row['rsi']:.1f}  @ {row['close']:.4f}  "
                  f"trend={'↑' if daily_uptrend else '↓'}  [{candle_time}]  → order placed ✓")
        except Exception as e:
            print(f"  [{pair}] ERROR placing buy: {e}")
    else:
        trend_str = "↑ uptrend" if daily_uptrend else "↓ downtrend (filter active)"
        print(f"  [{pair}] HOLD   rsi={row['rsi']:.1f}  close={row['close']:.4f}  "
              f"{trend_str}  [{candle_time}]")

    return state


# ── main loop ──────────────────────────────────────────────────────────

def run():
    state = load_state()

    account = trading_client.get_account()
    print(f"[INFO]  Strategy : S4 RSI Reversion + 1d trend filter")
    print(f"[INFO]  Pairs    : {', '.join(PAIRS)}")
    print(f"[INFO]  Signals  : RSI(14) < 30 entry  |  RSI > 70 exit  |  1d EMA20>50 filter")
    print(f"[INFO]  Alpaca paper — buying power: ${float(account.buying_power):,.2f}")
    print(f"[INFO]  Order size: ${ORDER_SIZE_USD} notional  |  checks every {CANDLE_MINUTES}min")
    print(f"[INFO]  Trades logged to {TRADES_FILE}\n")

    # reconcile state vs Alpaca on startup
    for pair in PAIRS:
        qty = get_qty(pair)
        if qty > 0 and pair in state:
            print(f"[RESUME] {pair} open LONG {qty} @ {state[pair]['entry_price']} "
                  f"entered {state[pair]['entry_time']}")
        elif qty == 0 and pair in state:
            print(f"[RESUME] {pair} stale state cleared (no Alpaca position)")
            state.pop(pair)

    def on_exit(sig, frame):
        save_state(state)
        open_pairs = [p for p in PAIRS if get_qty(p) > 0]
        if open_pairs:
            print(f"\n[STOP] Open positions on Alpaca: {', '.join(open_pairs)}")
        print("[STOP] Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    while True:
        wait = seconds_until_next_candle(CANDLE_MINUTES)
        next_time = datetime.now(timezone.utc) + timedelta(seconds=wait)
        print(f"\n[{now_str()}] Sleeping {wait:.0f}s → next check at "
              f"{next_time.strftime('%H:%M:%S UTC')}")
        time.sleep(wait)

        print(f"[{now_str()}] Checking {len(PAIRS)} pairs (S4 RSI)...")
        for pair in PAIRS:
            state = process_pair(pair, state)
        save_state(state)


if __name__ == "__main__":
    run()
