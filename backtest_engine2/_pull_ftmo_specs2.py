"""Fix pass: subscribe watchlist symbols (symbol_select) to get live prices, then decode
swap to bp/night and %/yr for ALL swap modes. Updates data/ftmo_specs.parquet prices."""
import sys

import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print(f"initialize() failed: {mt5.last_error()}")
    sys.exit(1)

OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_specs.parquet"
df = pd.read_parquet(OUT)

WATCH = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500.cash", "US100.cash", "US30.cash",
         "GER40.cash", "XAUUSD", "BTCUSD", "ETHUSD"]
for name in WATCH:
    mt5.symbol_select(name, True)

rows = []
for name in WATCH:
    s = mt5.symbol_info(name)
    t = mt5.symbol_info_tick(name)
    price = t.bid if t and t.bid > 0 else s.bid
    if price <= 0:  # market closed for this symbol right now -> last daily close
        r = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_D1, 0, 1)
        price = float(r[0]["close"]) if r is not None and len(r) else 0.0
    df.loc[df["symbol"] == name, "price"] = price
    mode = s.swap_mode
    if mode == 1:    # points: pts * point * contract_size = ccy/lot/night
        usd_night_long = s.swap_long * s.point * s.trade_contract_size
        usd_night_short = s.swap_short * s.point * s.trade_contract_size
        notional = s.trade_contract_size * price
        bp_l = 1e4 * usd_night_long / notional
        bp_s = 1e4 * usd_night_short / notional
    elif mode in (5, 6):  # interest: swap value = ANNUAL RATE IN %, charged daily
        bp_l = s.swap_long / 360 * 100      # %/yr -> bp/night
        bp_s = s.swap_short / 360 * 100
    else:
        bp_l = bp_s = float("nan")
    rows.append({"symbol": name, "mode": {1: "points", 5: "interest", 6: "interest"}.get(mode, mode),
                 "price": round(price, 2), "swap_long_raw": s.swap_long,
                 "swap_short_raw": s.swap_short, "3day": {3: "Wed", 5: "Fri"}.get(s.swap_rollover3days, s.swap_rollover3days),
                 "long_bp_nt": round(bp_l, 2), "long_pct_yr": round(bp_l * 364 / 100, 2),
                 "short_bp_nt": round(bp_s, 2), "short_pct_yr": round(bp_s * 364 / 100, 2),
                 "spread_bp": round(1e4 * s.spread * s.point / price, 2) if price > 0 else None})
mt5.shutdown()

df.to_parquet(OUT, index=False)
out = pd.DataFrame(rows)
print(out.to_string(index=False))
