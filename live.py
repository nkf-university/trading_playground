"""
Live paper trading loop — Bollinger Band Fade strategy → Alpaca paper account.

Run:   python live.py
Stop:  Ctrl+C  (open position persists on Alpaca; state file tracks entry metadata)

Signal source : Binance 5m candles via ccxt (public API)
Execution     : Alpaca paper trading account (real paper orders)

Note: Alpaca does not support crypto short selling.
  - LONG signal  → BUY on Alpaca
  - SHORT signal, no position → logged but skipped
  - SHORT signal, have position → treated as exit (SELL)
  - Middle-band exit → SELL full position
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
BINANCE_SYMBOL = "SOL/USDT"   # data source (Binance via ccxt)
ALPACA_SYMBOL  = "SOL/USD"    # execution (Alpaca)
TIMEFRAME      = "5m"
CANDLE_MINUTES = 5
ORDER_SIZE_USD = 10.0          # notional dollars per trade
TRADES_FILE    = "trades.csv"
STATE_FILE     = ".live_state.json"
# ───────────────────────────────────────────────────────────────────────

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)


# ── Alpaca helpers ─────────────────────────────────────────────────────

def get_alpaca_qty() -> float:
    """Return current SOL position qty on Alpaca (0.0 if none)."""
    try:
        pos = trading_client.get_open_position("SOLUSD")
        return float(pos.qty)
    except Exception:
        return 0.0


def alpaca_buy():
    order = MarketOrderRequest(
        symbol=ALPACA_SYMBOL,
        notional=ORDER_SIZE_USD,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )
    return trading_client.submit_order(order)


def alpaca_sell(qty: float):
    order = MarketOrderRequest(
        symbol=ALPACA_SYMBOL,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
    )
    return trading_client.submit_order(order)


# ── state persistence (entry metadata only — Alpaca owns position truth) ──

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"entry_price": None, "entry_time": None, "side": None}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


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
    return candle_minutes * 60 - seconds_past + 5  # +5s buffer after close


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── main loop ──────────────────────────────────────────────────────────

def run():
    state = load_state()

    # reconcile state file against actual Alpaca position on startup
    alpaca_qty = get_alpaca_qty()
    if alpaca_qty == 0 and state["entry_price"]:
        print("[RESUME] State file exists but no Alpaca position found — clearing stale state.")
        clear_state()
        state = {"entry_price": None, "entry_time": None, "side": None}
    elif alpaca_qty > 0 and state["entry_price"]:
        print(f"[RESUME] Open LONG {alpaca_qty} SOL @ {state['entry_price']} entered {state['entry_time']}")
    elif alpaca_qty > 0 and not state["entry_price"]:
        print(f"[RESUME] Alpaca shows open position but no local state — entry metadata unavailable.")

    account = trading_client.get_account()
    print(f"[INFO]  Strategy : Bollinger Band Fade  |  {BINANCE_SYMBOL}  |  {TIMEFRAME}")
    print(f"[INFO]  Alpaca paper account — buying power: ${float(account.buying_power):,.2f}")
    print(f"[INFO]  Order size: ${ORDER_SIZE_USD} notional per trade")
    print(f"[INFO]  Trades logged to {TRADES_FILE}")
    print(f"[INFO]  Press Ctrl+C to stop\n")

    def on_exit(sig, frame):
        qty = get_alpaca_qty()
        if qty > 0:
            print(f"\n[STOP] Open position of {qty} SOL remains on Alpaca paper account.")
        print("[STOP] Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    while True:
        wait = seconds_until_next_candle(CANDLE_MINUTES)
        next_time = datetime.now(timezone.utc) + timedelta(seconds=wait)
        print(f"[{now_str()}] Sleeping {wait:.0f}s → next check at {next_time.strftime('%H:%M:%S UTC')}")
        time.sleep(wait)

        # ── fetch + compute ────────────────────────────────────────────
        try:
            df = fetch_ohlcv(BINANCE_SYMBOL, TIMEFRAME, days=1)
            df = compute_indicators(df)
        except Exception as e:
            print(f"[{now_str()}] ERROR fetching data: {e} — skipping tick")
            continue

        if len(df) < 3:
            print(f"[{now_str()}] Not enough candles, skipping.")
            continue

        # last fully closed candle (second-to-last; last is still forming)
        row = df.iloc[-2]
        prev_row = df.iloc[-3]
        candle_time = row.name.strftime("%H:%M UTC")

        # check actual Alpaca position as source of truth
        alpaca_qty = get_alpaca_qty()
        in_position = alpaca_qty > 0

        # ── exit check ─────────────────────────────────────────────────
        if in_position and state["entry_price"]:
            # exit on middle band OR if a short signal fires while long
            sig = get_signal(row, prev_row)
            exit_triggered = should_exit(row, "long") or sig == "short"

            if exit_triggered:
                try:
                    alpaca_sell(alpaca_qty)
                    exit_price = row["close"]
                    entry_price = state["entry_price"]
                    entry_time = datetime.fromisoformat(state["entry_time"])
                    hold_minutes = (row.name.to_pydatetime() - entry_time).total_seconds() / 60
                    pnl_pct = (exit_price - entry_price) / entry_price * 100

                    trade = {
                        "timestamp": state["entry_time"],
                        "ticker": BINANCE_SYMBOL,
                        "side": "long",
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "pnl_pct": round(pnl_pct, 4),
                        "hold_time_minutes": round(hold_minutes, 1),
                    }
                    log_trade(trade)
                    reason = "middle band" if should_exit(row, "long") else "short signal"
                    print(f"[{now_str()}] EXIT  LONG   entry={entry_price:.4f}  exit={exit_price:.4f}  "
                          f"pnl={pnl_pct:+.3f}%  hold={hold_minutes:.0f}m  reason={reason}  [{candle_time}]")
                    clear_state()
                    state = {"entry_price": None, "entry_time": None, "side": None}
                except Exception as e:
                    print(f"[{now_str()}] ERROR placing sell order: {e}")
                continue

        # ── entry check ────────────────────────────────────────────────
        if not in_position:
            sig = get_signal(row, prev_row)

            if sig == "long":
                try:
                    alpaca_buy()
                    state = {
                        "entry_price": row["close"],
                        "entry_time": row.name.isoformat(),
                        "side": "long",
                    }
                    save_state(state)
                    print(f"[{now_str()}] ENTRY LONG   @ {row['close']:.4f}  "
                          f"rsi={row['rsi']:.1f}  vol_ratio={row['volume']/row['vol_ma']:.2f}x  "
                          f"[{candle_time}]  → Alpaca order placed ✓")
                except Exception as e:
                    print(f"[{now_str()}] ERROR placing buy order: {e}")

            elif sig == "short":
                print(f"[{now_str()}] SHORT signal @ {row['close']:.4f}  rsi={row['rsi']:.1f}  "
                      f"[{candle_time}]  → skipped (Alpaca crypto no shorting)")

            else:
                print(f"[{now_str()}] HOLD   rsi={row['rsi']:.1f}  "
                      f"close={row['close']:.4f}  bb_mid={row['bb_middle']:.4f}  [{candle_time}]")

        else:
            unrealised = (row["close"] - state["entry_price"]) / state["entry_price"] * 100 if state["entry_price"] else 0
            print(f"[{now_str()}] IN POSITION  qty={alpaca_qty:.6f} SOL  "
                  f"entry={state['entry_price']:.4f}  current={row['close']:.4f}  "
                  f"unrealised={unrealised:+.3f}%  mid={row['bb_middle']:.4f}  [{candle_time}]")


if __name__ == "__main__":
    run()
