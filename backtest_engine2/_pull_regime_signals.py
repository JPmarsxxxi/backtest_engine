# Pull leading regime signals (all free, keyless):
#  1. VIX + VIX3M  (CBOE daily CSV)      -> fear-gauge term structure
#  2. HY OAS       (FRED BAMLH0A0HYM2)   -> high-yield credit spread (credit cracks first)
#  3. NFCI         (FRED NFCI)           -> Chicago Fed financial-conditions index
# Saved to data/regime_signals.parquet (daily, forward-filled weekly NFCI).
import io, sys
import pandas as pd
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research data pull)"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")

def cboe(sym):
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    df = pd.read_csv(io.StringIO(get(url)))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    s = df.set_index("DATE")["CLOSE"].sort_index()
    s.name = sym
    return s

def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(io.StringIO(get(url)))
    dc, vc = df.columns[0], df.columns[1]
    df[dc] = pd.to_datetime(df[dc])
    df[vc] = pd.to_numeric(df[vc], errors="coerce")
    s = df.set_index(dc)[vc].dropna().sort_index()
    s.name = series
    return s

out = {}
for label, fn in [("VIX", lambda: cboe("VIX")), ("VIX3M", lambda: cboe("VIX3M")),
                   ("HY_OAS", lambda: fred("BAMLH0A0HYM2")), ("NFCI", lambda: fred("NFCI"))]:
    try:
        s = fn()
        out[label] = s
        print(f"OK  {label:8s} {len(s):5d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"ERR {label:8s} {type(e).__name__}: {e}")

if not out:
    print("nothing pulled"); sys.exit(1)

# daily calendar union, weekly NFCI forward-filled to daily
panel = pd.concat(out, axis=1)
panel = panel.sort_index()
panel["NFCI"] = panel["NFCI"].ffill()           # weekly -> daily
if "VIX" in panel and "VIX3M" in panel:
    panel["VIX_TS"] = panel["VIX"] / panel["VIX3M"]   # >1 = backwardation = acute stress
path = r"C:\Users\User\backtest_engine\backtest_engine2\data\regime_signals.parquet"
panel.to_parquet(path)
print(f"\nsaved {panel.shape} -> {path}")
print("cols:", list(panel.columns))
print(panel.tail(3))
