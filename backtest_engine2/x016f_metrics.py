# #016 full metrics — combined 3-pair book + per-pair. Sizing = 1x notional per leg (the +12.5%/yr base);
# Sharpe/Sortino/Calmar are scale-invariant, maxDD/vol/worst-day scale with sizing.
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
HALF = {"EURUSD": {21: 0.087, 23: 0.086}, "GBPUSD": {21: 0.187, 23: 0.187},
        "USDJPY": {21: 0.156, 23: 0.125}}
SIGN = {21: -1.0, 23: +1.0}

def book(sym):
    s = pd.read_parquet(rf"{ENG}\data\{sym.lower()}_h1.parquet").set_index("open_time")["close"].sort_index()
    r = s.pct_change()
    r = r[(s.index.to_series().diff() == pd.Timedelta("1h")).values]
    ldn = r.index.tz_convert("Europe/London")
    d = pd.DataFrame({"ret": r.values, "hr": ldn.hour, "yr": ldn.year}, index=r.index)
    d["day"] = pd.to_datetime(pd.Series(ldn.date, index=d.index))
    d = d[(d.yr >= 2022) & d.hr.isin([21, 23])].copy()
    d["net"] = [SIGN[h] for h in d.hr] * d["ret"] - [2 * HALF[sym][h] / 1e4 for h in d.hr]
    return d.groupby("day")["net"].sum()

books = {s: book(s) for s in ["EURUSD", "GBPUSD", "USDJPY"]}
comb = pd.concat(books.values(), axis=1).dropna().sum(axis=1)

def metrics(daily, label):
    daily = daily.dropna()
    yrs = (daily.index[-1] - daily.index[0]).days / 365.25
    ppy = len(daily) / yrs
    annf = np.sqrt(ppy)
    eq = (1 + daily).cumprod()
    dd = eq / eq.cummax() - 1
    ann_ret = daily.mean() * ppy
    ann_vol = daily.std() * annf
    downside = daily[daily < 0].std() * annf
    sharpe = daily.mean() / daily.std() * annf
    sortino = ann_ret / downside
    maxdd = dd.min()
    calmar = ann_ret / abs(maxdd)
    # longest underwater stretch (days)
    uw = (dd < 0).astype(int)
    grp = (uw.diff() != 0).cumsum()
    longest = uw.groupby(grp).sum().max()
    wins, losses = daily[daily > 0], daily[daily < 0]
    print(f"\n===== {label} =====")
    print(f"  period            {daily.index[0].date()} -> {daily.index[-1].date()}  ({yrs:.1f}y, {ppy:.0f} days/yr)")
    print(f"  net Sharpe        {sharpe:+.2f}")
    print(f"  net Sortino       {sortino:+.2f}")
    print(f"  Calmar (ret/DD)   {calmar:+.2f}")
    print(f"  ann return        {ann_ret*100:+.1f}%        ann vol {ann_vol*100:.1f}%")
    print(f"  MAX DRAWDOWN      {maxdd*100:+.1f}%        longest underwater {int(longest)} days")
    print(f"  avg/day           {1e4*daily.mean():+.2f} bp     median {1e4*daily.median():+.2f} bp")
    print(f"  worst day         {1e4*daily.min():+.1f} bp ({daily.min()*100:+.2f}%)   best day {1e4*daily.max():+.1f} bp")
    print(f"  hit rate          {100*(daily>0).mean():.1f}%       win/loss bp {1e4*wins.mean():+.1f}/{1e4*losses.mean():+.1f}")
    print(f"  total net (cumul) {(eq.iloc[-1]-1)*100:+.1f}% over {yrs:.1f}y   final equity x{eq.iloc[-1]:.2f}")
    print(f"  FTMO check: worst day {daily.min()*100:+.2f}% vs -5% limit | maxDD {maxdd*100:+.1f}% vs -10% limit")

for s, b in books.items():
    metrics(b, f"{s} (1 pair, 2 hrs/day)")
metrics(comb, "COMBINED 3-PAIR BOOK (6 trades/day)")
