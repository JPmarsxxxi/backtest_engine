# Retry FRED pulls (HY OAS + NFCI), merge with VIX/VIX3M already pulled from CBOE.
import io, time
import pandas as pd
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research data pull)"}

def get(url, tries=4, timeout=120):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e; print(f"  attempt {k+1} failed: {type(e).__name__}; retrying...")
            time.sleep(3)
    raise last

def fred(series):
    for host in ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=",
                 "https://fred.stlouisfed.org/series/{}/downloaddata/{}.csv"):
        try:
            url = host + series if "?id=" in host else host.format(series, series)
            df = pd.read_csv(io.StringIO(get(url)))
            dc, vc = df.columns[0], df.columns[1]
            df[dc] = pd.to_datetime(df[dc]); df[vc] = pd.to_numeric(df[vc], errors="coerce")
            s = df.set_index(dc)[vc].dropna().sort_index(); s.name = series
            return s
        except Exception as e:
            print(f"  host failed for {series}: {type(e).__name__}")
    raise RuntimeError(f"all hosts failed for {series}")

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
panel = pd.read_parquet(DATA + r"\regime_signals.parquet") if False else None  # not saved last run
# VIX/VIX3M were pulled but not persisted; re-pull (fast) from CBOE
def cboe(sym):
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    df = pd.read_csv(io.StringIO(get(url, timeout=60)))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    s = df.set_index("DATE")["CLOSE"].sort_index(); s.name = sym; return s

out = {}
for label, fn in [("VIX", lambda: cboe("VIX")), ("VIX3M", lambda: cboe("VIX3M")),
                   ("HY_OAS", lambda: fred("BAMLH0A0HYM2")), ("NFCI", lambda: fred("NFCI"))]:
    try:
        s = fn(); out[label] = s
        print(f"OK  {label:8s} {len(s):5d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"ERR {label:8s} {type(e).__name__}: {e}")

panel = pd.concat(out, axis=1).sort_index()
if "NFCI" in panel: panel["NFCI"] = panel["NFCI"].ffill()
if "HY_OAS" in panel: panel["HY_OAS"] = panel["HY_OAS"].ffill()
if "VIX" in panel and "VIX3M" in panel: panel["VIX_TS"] = panel["VIX"] / panel["VIX3M"]
path = DATA + r"\regime_signals.parquet"
panel.to_parquet(path)
print(f"\nsaved {panel.shape} -> {path}\ncols: {list(panel.columns)}")
print(panel.dropna(how="all").tail(3))
