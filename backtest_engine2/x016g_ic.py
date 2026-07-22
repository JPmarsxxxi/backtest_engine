# #016 IC decomposition: IR = IC * sqrt(breadth). Signal is +/-1 per (hour), so per-trade
# IC = corr(sign, ret) = mean(signed)/std(signed) = per-trade Sharpe. Compare raw vs effective breadth
# (EUR/GBP +0.74 correlated -> effective breadth < raw count).
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
HALF = {"EURUSD": {21: 0.087, 23: 0.086}, "GBPUSD": {21: 0.187, 23: 0.187},
        "USDJPY": {21: 0.156, 23: 0.125}}
SIGN = {21: -1.0, 23: +1.0}

trades = []          # pooled individual (pair,hour,day) trades
daily_pair = {}
for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
    s = pd.read_parquet(rf"{ENG}\data\{sym.lower()}_h1.parquet").set_index("open_time")["close"].sort_index()
    r = s.pct_change()
    r = r[(s.index.to_series().diff() == pd.Timedelta("1h")).values]
    ldn = r.index.tz_convert("Europe/London")
    d = pd.DataFrame({"ret": r.values, "hr": ldn.hour, "yr": ldn.year}, index=r.index)
    d["day"] = pd.to_datetime(pd.Series(ldn.date, index=d.index))
    d = d[(d.yr >= 2022) & d.hr.isin([21, 23])].copy()
    d["sign"] = [SIGN[h] for h in d.hr]
    d["gross"] = d["sign"] * d["ret"]
    d["net"] = d["gross"] - [2 * HALF[sym][h] / 1e4 for h in d.hr]
    trades.append(d.assign(pair=sym))
    daily_pair[sym] = d.groupby("day")["net"].sum()

T = pd.concat(trades)
yrs = (T.index.max() - T.index.min()).days / 365.25

# per-trade IC (gross and net)
def per_trade(col):
    x = T[col]
    ic = T["sign"].corr(T["ret"]) if col == "gross" else np.nan
    sh = x.mean() / x.std()                              # = IC for +/-1 signal
    return sh
ic_gross = T["sign"].corr(T["ret"])
ic_rank = T["sign"].corr(T["ret"], method="spearman")
ptsh_gross = T["gross"].mean() / T["gross"].std()
ptsh_net = T["net"].mean() / T["net"].std()

raw_breadth = len(T) / yrs                                # bets/yr
print(f"pooled trades {len(T):,} over {yrs:.1f}y  ->  raw breadth {raw_breadth:.0f} bets/yr")
print(f"\nPER-TRADE IC (skill per bet):")
print(f"  IC = corr(sign, ret)        {ic_gross:+.4f}   (rank-IC {ic_rank:+.4f})")
print(f"  per-trade Sharpe gross      {ptsh_gross:+.4f}   net {ptsh_net:+.4f}   (== IC for +/-1 signal)")
print(f"  hit rate {100*(T['net']>0).mean():.1f}%   (rule-of-thumb IC~2*hit-1 = {2*(T['net']>0).mean()-1:+.3f})")

# per (pair,hour) IC
print(f"\n  IC by cell (per-trade Sharpe, net):")
for (p, h), g in T.groupby(["pair", "hr"]):
    print(f"    {p} {h:02d}:00  net per-trade Sh {g['net'].mean()/g['net'].std():+.3f}  (n={len(g)})")

# IR = IC * sqrt(breadth) reconciliation (combined book)
comb = pd.concat(daily_pair.values(), axis=1).dropna().sum(axis=1)
ppy = len(comb) / yrs
ir_real = comb.mean() / comb.std() * np.sqrt(ppy)        # realized annualized IR (net)
eff_breadth = (ir_real / ptsh_net) ** 2                  # effective independent bets/yr
naive_ir = ptsh_net * np.sqrt(raw_breadth)
print(f"\nIR = IC * sqrt(breadth):")
print(f"  realized annualized IR (combined, net)   {ir_real:+.2f}")
print(f"  naive IR = IC_net * sqrt(raw breadth)     {naive_ir:+.2f}   (assumes independent bets)")
print(f"  -> EFFECTIVE breadth = (IR/IC)^2          {eff_breadth:.0f} bets/yr  "
      f"(vs raw {raw_breadth:.0f}; ratio {eff_breadth/raw_breadth:.2f} = correlation haircut)")
print(f"  cross-pair corr drag: EUR-GBP +0.74 shares bets; USDJPY -0.4 adds them back")
