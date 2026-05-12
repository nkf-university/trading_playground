"""
One-shot test: places a $10 BTC/USD paper buy order then immediately sells it.
Run once to verify Railway → Alpaca connection is working.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config
import time

client = TradingClient(config.API_KEY, config.SECRET_KEY, paper=True)

acc = client.get_account()
print(f"Account: {acc.status}  |  trading_blocked={acc.trading_blocked}  |  buying_power=${float(acc.buying_power):,.2f}")

print("Placing test BUY $10 BTC/USD...")
buy = client.submit_order(MarketOrderRequest(
    symbol="BTC/USD", notional=10, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
))
print(f"BUY order placed — id={buy.id}  status={buy.status}")

time.sleep(3)

print("Placing test SELL...")
pos = client.get_open_position("BTCUSD")
sell = client.submit_order(MarketOrderRequest(
    symbol="BTC/USD", qty=float(pos.qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC
))
print(f"SELL order placed — id={sell.id}  status={sell.status}")
print("Test complete — check Alpaca dashboard for both orders.")
