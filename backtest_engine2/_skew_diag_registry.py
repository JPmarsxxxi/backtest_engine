# diagnostic: map trial_store returns files to strategies
import sqlite3, pandas as pd
con = sqlite3.connect(r"C:\Users\User\backtest_engine\backtest_engine2\trial_store\registry.sqlite")
tabs = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
print(tabs)
for t in tabs["name"]:
    df = pd.read_sql(f"SELECT * FROM {t}", con)
    print(f"--- {t} ({len(df)} rows) ---")
    print(df.to_string(max_colwidth=60))
con.close()
