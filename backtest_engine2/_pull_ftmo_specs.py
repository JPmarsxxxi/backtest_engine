"""Dump FTMO MT5 contract specs (swap, contract size, digits, spread) for ALL symbols.
Attaches to a RUNNING, LOGGED-IN MT5 terminal. Output: data/ftmo_specs.parquet + console table.
Swap conversion: bp/night = swap_pts * point * contract_size / (price_usd_notional) * 1e4,
%/yr ~= bp/night * 3.64 (nightly charge, 3x one day a week -> ~364 charges/yr)."""
import sys

import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print(f"initialize() failed: {mt5.last_error()}")
    print("-> Is the FTMO MT5 terminal running and logged in?")
    sys.exit(1)

acct = mt5.account_info()
print(f"Connected: {acct.server} | login {acct.login} | {acct.company} | "
      f"balance {acct.balance:,.0f} {acct.currency}\n")

SWAP_MODE = {0: "disabled", 1: "points", 2: "base_ccy", 3: "margin_ccy", 4: "deposit_ccy",
             5: "interest_current", 6: "interest_open", 7: "reopen_current", 8: "reopen_bid"}
DAYS = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}

rows = []
for s in mt5.symbols_get():
    tick = mt5.symbol_info_tick(s.name)
    price = tick.bid if tick and tick.bid > 0 else s.bid
    rows.append({
        "symbol": s.name,
        "path": s.path,                      # asset-class folder, e.g. "Forex\\Majors\\EURUSD"
        "digits": s.digits,
        "point": s.point,
        "contract_size": s.trade_contract_size,
        "swap_mode": SWAP_MODE.get(s.swap_mode, str(s.swap_mode)),
        "swap_long": s.swap_long,
        "swap_short": s.swap_short,
        "swap_3day": DAYS.get(s.swap_rollover3days, str(s.swap_rollover3days)),
        "spread_pts": s.spread,
        "price": price,
        "currency_profit": s.currency_profit,
        "trade_allowed": s.trade_mode != 0,
    })
mt5.shutdown()

df = pd.DataFrame(rows)
OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_specs.parquet"
df.to_parquet(OUT, index=False)
print(f"{len(df)} symbols -> {OUT}")
print(f"asset classes: {df['path'].str.split('\\\\').str[0].value_counts().to_dict()}\n")

# Decoded swap preview for the instruments that gate the queue (points-mode only here;
# other modes decoded in the analysis cell).
WATCH = ["US500", "US100", "US30", "GER40", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY",
         "AUDUSD", "BTCUSD", "ETHUSD"]
w = df[df["symbol"].str.upper().str.replace(".CASH", "", regex=False).isin(WATCH)].copy()
if len(w):
    usd_lot = w["contract_size"] * w["price"]  # rough notional (USD-quoted instruments)
    for side in ("long", "short"):
        usd_night = w[f"swap_{side}"] * w["point"] * w["contract_size"]
        w[f"{side}_bp_nt"] = (1e4 * usd_night / usd_lot).round(2)
        w[f"{side}_pct_yr"] = (w[f"{side}_bp_nt"] * 364 / 100).round(2)
    cols = ["symbol", "swap_mode", "swap_long", "swap_short", "swap_3day",
            "long_bp_nt", "long_pct_yr", "short_bp_nt", "short_pct_yr", "spread_pts"]
    print(w[cols].to_string(index=False))
else:
    print("(none of the watchlist symbols found by exact name — check df['symbol'] naming)")
