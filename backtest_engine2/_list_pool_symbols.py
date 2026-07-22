import pandas as pd

df = pd.read_parquet(
    r"C:\Users\User\backtest_engine\backtest_engine2\data\binance_hourly_top50_pool.parquet",
    columns=["symbol"],
)
syms = sorted(df["symbol"].unique())
print(len(syms))
print(",".join(syms))
