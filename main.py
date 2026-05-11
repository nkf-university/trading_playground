import csv
import os
import pandas as pd

from data import fetch_ohlcv
from strategy import compute_indicators, get_signal, should_exit

SYMBOL = "SOL/USD"
TIMEFRAME = "5m"
DAYS = 30
TRADES_FILE = "trades.csv"


def run_backtest(df: pd.DataFrame) -> list:
    trades = []
    position = None       # None | 'long' | 'short'
    entry_price = 0.0
    entry_time = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # ── exit check (always before entry to avoid same-bar entry+exit) ──
        if position is not None:
            if should_exit(row, position):
                exit_price = row["close"]
                hold_minutes = (row.name - entry_time).total_seconds() / 60
                pnl_pct = (
                    (exit_price - entry_price) / entry_price * 100
                    if position == "long"
                    else (entry_price - exit_price) / entry_price * 100
                )
                trades.append({
                    "timestamp": entry_time.isoformat(),
                    "ticker": SYMBOL,
                    "side": position,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "hold_time_minutes": round(hold_minutes, 1),
                })
                print(
                    f"  EXIT  {position:<5}  entry={entry_price:.4f}  "
                    f"exit={exit_price:.4f}  pnl={pnl_pct:+.3f}%  "
                    f"hold={hold_minutes:.0f}m"
                )
                position = None

        # ── entry check (only when flat — no pyramiding) ───────────────────
        if position is None:
            signal = get_signal(row, prev_row)
            if signal:
                position = signal
                entry_price = row["close"]
                entry_time = row.name
                print(f"  ENTRY {signal:<5}  @ {entry_price:.4f}  rsi={row['rsi']:.1f}  {row.name}")

    # close any position still open at end of data
    if position is not None:
        exit_price = df.iloc[-1]["close"]
        hold_minutes = (df.iloc[-1].name - entry_time).total_seconds() / 60
        pnl_pct = (
            (exit_price - entry_price) / entry_price * 100
            if position == "long"
            else (entry_price - exit_price) / entry_price * 100
        )
        trades.append({
            "timestamp": entry_time.isoformat(),
            "ticker": SYMBOL,
            "side": position,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "pnl_pct": round(pnl_pct, 4),
            "hold_time_minutes": round(hold_minutes, 1),
        })

    return trades


def save_trades(trades: list):
    if not trades:
        return
    fieldnames = ["timestamp", "ticker", "side", "entry_price", "exit_price", "pnl_pct", "hold_time_minutes"]
    write_header = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(trades)


def print_summary(trades: list):
    if not trades:
        print("No trades executed.")
        return

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    total = len(pnls)
    win_rate = len(wins) / total * 100
    avg_pnl = sum(pnls) / total

    # max drawdown on compounded equity curve
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for p in pnls:
        equity *= 1 + p / 100
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    longs = [t for t in trades if t["side"] == "long"]
    shorts = [t for t in trades if t["side"] == "short"]

    print("\n" + "=" * 48)
    print(f"  BACKTEST SUMMARY  —  {SYMBOL}  ({DAYS}d, {TIMEFRAME})")
    print("=" * 48)
    print(f"  Total trades     : {total}  (long={len(longs)}, short={len(shorts)})")
    print(f"  Win rate         : {win_rate:.1f}%")
    print(f"  Avg P&L / trade  : {avg_pnl:+.3f}%")
    print(f"  Total return     : {(equity - 1) * 100:+.2f}%")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print("=" * 48)

    print(f"\n  {'Timestamp':<20} {'Side':<6} {'Entry':>9} {'Exit':>9} {'P&L%':>8} {'Hold(m)':>8}")
    print("  " + "-" * 64)
    for t in trades:
        ts = t["timestamp"][:16].replace("T", " ")
        pnl_str = f"{t['pnl_pct']:+.3f}%"
        print(f"  {ts:<20} {t['side']:<6} {t['entry_price']:>9.4f} {t['exit_price']:>9.4f} {pnl_str:>8} {t['hold_time_minutes']:>8.1f}")


if __name__ == "__main__":
    print(f"Fetching {DAYS}d of {TIMEFRAME} candles for {SYMBOL} from Binance...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, DAYS)
    print(f"  {len(df):,} candles  ({df.index[0]}  →  {df.index[-1]})\n")

    print("Computing indicators (BB/RSI/volume)...")
    df = compute_indicators(df)

    print(f"\nRunning backtest...\n")
    trades = run_backtest(df)

    save_trades(trades)
    print(f"\n  → {len(trades)} trades saved to {TRADES_FILE}")

    print_summary(trades)
