# Cell 2 — Stage 1 amendment: rebuild panel from the gap-repaired parquet.
# (Puller's vol>0 filter had dropped 2022-07-26..2023-05-19 — volume field was zeroed
# server-side in that window. Re-pulled with price-change-based activity filter, merged.)
# Outliers (COVID March 2020) reviewed and accepted in Cell 1 -> check_outliers=False.
RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet")
s = RAW.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)

px30 = s.resample("30min", closed="left", label="right").last().dropna()
prices = px30.to_frame("USA500")

panel = DataPanel(prices, check_outliers=False)

print(f"Bars: {len(panel.dates)}")
print(f"Range: {panel.dates[0]} -> {panel.dates[-1]}")
print("Bars per year:", prices.groupby(prices.index.year).size().to_dict())
bars_per_day = prices.groupby(prices.index.normalize()).size()
print(f"Days with data: {len(bars_per_day)}; bars/day median: {bars_per_day.median():.0f}")
prices.tail(3)
