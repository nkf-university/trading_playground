"""
Live paper trading loop — Bollinger Band Fade strategy → Alpaca paper account.

Run:   python live.py
Stop:  Ctrl+C

Trades all 8 pairs independently. Each pair has its own position on Alpaca.
Signal source : Alpaca crypto data (5m candles)
Execution     : Alpaca paper trading (real paper orders)

Note: Alpaca does not support crypto short selling.
  - LONG signal             → BUY on Alpaca
  - SHORT signal, no pos    → skipped
  - SHORT signal, have pos  → treated as exit (SELL)
  - Middle-band cross       → SELL full position
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
from data import fetch_ohlcv
from strategy import compute_indicators, get_signal, should_exit

# ── config ─────────────────────────────────────────────────────────────
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LINK/USD", "AAVE/USD", "DOT/USD"]
TIMEFRAME      = "5m"
CANDLE_MINUTES = 5
ORDER_SIZE_USD = 10.0
TRADES_FILE    = "trades.csv"
STATE_FILE     = ".live_state.json"
# ───────────────────────────────────────────────────────────────────────

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)


# ── Alpaca helpers ─────────────────────────────────────────────────────

def alpaca_symbol(pair: str) -> str:
    """SOL/USD → SOLUSD (used for position lookups)."""
    return pair.replace("/", "")


def get_qty(pair: str) -> float:
    try:
        pos = trading_client.get_open_position(alpaca_symbol(pair))
        return float(pos.qty)
    except Exception:
        return 0.0


def place_buy(pair: str):
    order = MarketOrderRequest(
        symbol=pair,
        notional=ORDER_SIZE_USD,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )
    return trading_client.submit_order(order)


def place_sell(pair: str, qty: float):
    order = MarketOrderRequest(
        symbol=pair,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
    )
    return trading_client.submit_order(order)


# ── state (entry metadata per pair) ───────────────────────────────────

def load_state() -> dict:
    """Returns dict keyed by pair, e.g. {'SOL/USD': {entry_price, entry_time}}"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ── trade logging ──────────────────────────────────────────────────────

def log_trade(trade: dict):
    fieldnames = ["timestamp", "ticker", "side", "entry_price", "exit_price", "pnl_pct", "hold_time_minutes"]
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
    return candle_minutes * 60 - seconds_past + 5


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── per-pair tick ──────────────────────────────────────────────────────

def process_pair(pair: str, state: dict) -> dict:
    try:
        df = fetch_ohlcv(pair, TIMEFRAME, days=1)
        df = compute_indicators(df)
    except Exception as e:
        print(f"  [{pair}] ERROR fetching data: {e}")
        return state

    if len(df) < 3:
        return state

    row = df.iloc[-2]       # last closed candle
    prev_row = df.iloc[-3]
    candle_time = row.name.strftime("%H:%M UTC")

    qty = get_qty(pair)
    in_position = qty > 0
    pair_state = state.get(pair)

    # ── exit ──────────────────────────────────────────────────────────
    if in_position and pair_state:
        sig = get_signal(row, prev_row)
        exit_triggered = should_exit(row, "long") or sig == "short"

        if exit_triggered:
            try:
                place_sell(pair, qty)
                entry_price = pair_state["entry_price"]
                entry_time = datetime.fromisoformat(pair_state["entry_time"])
                exit_price = row["close"]
                hold_minutes = (row.name.to_pydatetime() - entry_time).total_seconds() / 60
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                reason = "mid-band" if should_exit(row, "long") else "short signal"

                log_trade({
                    "timestamp": pair_state["entry_time"],
                    "ticker": pair,
                    "side": "long",
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "pnl_pct": round(pnl_pct, 4),
                    "hold_time_minutes": round(hold_minutes, 1),
                })
                print(f"  [{pair}] EXIT  entry={entry_price:.4f}  exit={exit_price:.4f}  "
                      f"pnl={pnl_pct:+.3f}%  hold={hold_minutes:.0f}m  reason={reason}  [{candle_time}]")
                state.pop(pair, None)
            except Exception as e:
                print(f"  [{pair}] ERROR placing sell: {e}")
            return state

    # ── entry ─────────────────────────────────────────────────────────
    if not in_position:
        sig = get_signal(row, prev_row)

        if sig == "long":
            try:
                place_buy(pair)
                state[pair] = {
                    "entry_price": row["close"],
                    "entry_time": row.name.isoformat(),
                }
                print(f"  [{pair}] ENTRY LONG  @ {row['close']:.4f}  "
                      f"rsi={row['rsi']:.1f}  vol={row['volume']/row['vol_ma']:.2f}x  "
                      f"[{candle_time}]  → order placed ✓")
            except Exception as e:
                print(f"  [{pair}] ERROR placing buy: {e}")

        elif sig == "short":
            print(f"  [{pair}] SHORT skipped  rsi={row['rsi']:.1f}  [{candle_time}]")

        else:
            unrealised_str = ""
            if in_position and pair_state:
                unr = (row["close"] - pair_state["entry_price"]) / pair_state["entry_price"] * 100
                unrealised_str = f"  unrealised={unr:+.3f}%"
            print(f"  [{pair}] HOLD  rsi={row['rsi']:.1f}  close={row['close']:.4f}{unrealised_str}  [{candle_time}]")
    else:
        unr = (row["close"] - pair_state["entry_price"]) / pair_state["entry_price"] * 100 if pair_state else 0
        print(f"  [{pair}] IN POSITION  qty={qty:.6f}  entry={pair_state['entry_price']:.4f}  "
              f"current={row['close']:.4f}  unrealised={unr:+.3f}%  mid={row['bb_middle']:.4f}  [{candle_time}]")

    return state


# ── main loop ──────────────────────────────────────────────────────────

def run():
    state = load_state()

    account = trading_client.get_account()
    print(f"[INFO]  Strategy : Bollinger Band Fade  |  {len(PAIRS)} pairs  |  {TIMEFRAME} candles")
    print(f"[INFO]  Pairs    : {', '.join(PAIRS)}")
    print(f"[INFO]  Alpaca paper account — buying power: ${float(account.buying_power):,.2f}")
    print(f"[INFO]  Order size: ${ORDER_SIZE_USD} notional per trade  |  max exposure: ${ORDER_SIZE_USD * len(PAIRS):.0f}")
    print(f"[INFO]  Trades logged to {TRADES_FILE}")
    print(f"[INFO]  Press Ctrl+C to stop\n")

    # reconcile state against actual Alpaca positions on startup
    for pair in PAIRS:
        qty = get_qty(pair)
        if qty > 0 and pair not in state:
            print(f"[RESUME] {pair} has open position ({qty}) but no local state — entry metadata unavailable")
        elif qty == 0 and pair in state:
            print(f"[RESUME] {pair} state file exists but no Alpaca position — clearing stale state")
            state.pop(pair)
        elif qty > 0 and pair in state:
            print(f"[RESUME] {pair} open LONG {qty} @ {state[pair]['entry_price']} entered {state[pair]['entry_time']}")

    def on_exit(sig, frame):
        save_state(state)
        open_pairs = [p for p in PAIRS if get_qty(p) > 0]
        if open_pairs:
            print(f"\n[STOP] Open positions remain on Alpaca: {', '.join(open_pairs)}")
        print("[STOP] Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    while True:
        wait = seconds_until_next_candle(CANDLE_MINUTES)
        next_time = datetime.now(timezone.utc) + timedelta(seconds=wait)
        print(f"\n[{now_str()}] Sleeping {wait:.0f}s → next check at {next_time.strftime('%H:%M:%S UTC')}")
        time.sleep(wait)
        print(f"[{now_str()}] Checking {len(PAIRS)} pairs...")

        for pair in PAIRS:
            state = process_pair(pair, state)

        save_state(state)


if __name__ == "__main__":
    run()
