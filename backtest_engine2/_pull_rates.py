# Option B: long-history macro regime data from Yahoo (keyless).
#  ^TNX = 10y Treasury yield, ^IRX = 13-week (3mo) T-bill -> yield-curve spread (10y-3mo).
#  ^GSPC extended back to ~1950 for long price-trend history.
# Saved to data/rates.parquet (+ refresh long SPX into data/spx_long.parquet).
import json, time
import pandas as pd
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 research"}

def get(url, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e; print(f"  attempt {k+1} {type(e).__name__}; retry..."); time.sleep(2)
    raise last

def yahoo(sym, period1="-631152000"):   # default ~1950
    for host in ("query1", "query2"):
        try:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
                   f"?period1={period1}&period2=9999999999&interval=1d")
            j = json.loads(get(url))
            res = j["chart"]["result"][0]
            ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
            close = res["indicators"]["quote"][0]["close"]
            s = pd.Series(close, index=ts).dropna(); s.name = sym
            return s
        except Exception as e:
            print(f"  {host} failed for {sym}: {type(e).__name__}: {str(e)[:60]}")
    raise RuntimeError(f"yahoo failed for {sym}")

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
out = {}
for label, sym in [("TNX10Y", "%5ETNX"), ("IRX3M", "%5EIRX"), ("SPX", "%5EGSPC")]:
    try:
        s = yahoo(sym); out[label] = s
        print(f"OK  {label:8s} {len(s):6d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"ERR {label:8s} {e}")

panel = pd.concat(out, axis=1).sort_index()
if "TNX10Y" in panel and "IRX3M" in panel:
    panel["YC_SPREAD"] = panel["TNX10Y"] - panel["IRX3M"]   # >0 normal, <0 inverted = danger
panel.to_parquet(DATA + r"\rates.parquet")
print(f"\nsaved {panel.shape} -> data/rates.parquet  cols: {list(panel.columns)}")
print("yield-curve spread coverage:", panel['YC_SPREAD'].dropna().index.min().date(),
      "->", panel['YC_SPREAD'].dropna().index.max().date())
print(panel.dropna(subset=['YC_SPREAD']).tail(2))
