import pandas as pd
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from strategies.crypto_rsi import rsi_signal

# --- Config -----------------------------------------------------------
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LINK/USD", "AAVE/USD", "DOT/USD"]
ORDER_SIZE_USD = 10.0            # notional dollars per order (keep small)
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
BAR_HOURS = 72                   # lookback window in hours
# ----------------------------------------------------------------------

trading_client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient(config.API_KEY, config.SECRET_KEY)


def get_bars(symbol: str) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=BAR_HOURS)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
    )
    bars = data_client.get_crypto_bars(request).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars.sort_index()


def get_position_qty(symbol: str) -> float:
    # Alpaca stores crypto positions without the slash: "BTCUSD"
    alpaca_symbol = symbol.replace("/", "")
    try:
        position = trading_client.get_open_position(alpaca_symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def place_order(symbol: str, side: OrderSide):
    order = MarketOrderRequest(
        symbol=symbol,
        notional=ORDER_SIZE_USD,  # dollar-based sizing
        side=side,
        time_in_force=TimeInForce.GTC,
    )
    result = trading_client.submit_order(order)
    print(f"  [ORDER] {side.value.upper()} ${ORDER_SIZE_USD} of {symbol} — id: {result.id}")
    return result


def run():
    print(f"[{datetime.now().isoformat()}] Crypto RSI strategy")

    account = trading_client.get_account()
    print(f"[ACCOUNT] Buying power: ${float(account.buying_power):,.2f}\n")

    for pair in PAIRS:
        print(f"--- {pair} ---")
        try:
            bars = get_bars(pair)
        except Exception as e:
            print(f"  [ERROR] Failed to fetch bars: {e}")
            continue

        signal, rsi_val = rsi_signal(bars, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT)
        position_qty = get_position_qty(pair)

        print(f"  RSI({RSI_PERIOD}): {rsi_val:.1f}  |  signal: {signal.upper()}  |  position: {position_qty}")

        if signal == "buy" and position_qty == 0:
            place_order(pair, OrderSide.BUY)
        elif signal == "sell" and position_qty > 0:
            # sell entire position
            alpaca_symbol = pair.replace("/", "")
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=position_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            result = trading_client.submit_order(order)
            print(f"  [ORDER] SELL {position_qty} {pair} — id: {result.id}")
        else:
            print(f"  [SKIP] No action")

        print()


if __name__ == "__main__":
    run()
