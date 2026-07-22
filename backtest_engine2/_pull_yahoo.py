# FRED + Stooq blocked. Try Yahoo Finance chart JSON (keyless).
#  HYG/IEF = credit-stress proxy (incl. 2008); ^GSPC = long S&P history (incl. 2000 + 2008 bears).
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

def yahoo(sym):
    for host in ("query1", "query2"):
        try:
            url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}"
                   f"?period1=315532800&period2=9999999999&interval=1d")  # 1980->now, DAILY
            j = json.loads(get(url))
            res = j["chart"]["result"][0]
            ts = pd.to_datetime(res["timestamp"], unit="s")
            q = res["indicators"]["quote"][0]["close"]
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q)
            s = pd.Series(adj, index=ts).dropna(); s.name = sym
            s.index = s.index.normalize()
            return s
        except Exception as e:
            print(f"  {host} failed for {sym}: {type(e).__name__}: {str(e)[:60]}")
    raise RuntimeError(f"yahoo failed for {sym}")

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
out = {}
for label, sym in [("HYG", "HYG"), ("IEF", "IEF"), ("SPX", "%5EGSPC")]:
    try:
        s = yahoo(sym); out[label] = s
        print(f"OK  {label:5s} {len(s):6d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"ERR {label:5s} {e}")

if out:
    panel = pd.concat(out, axis=1).sort_index()
    if "HYG" in panel and "IEF" in panel:
        panel["CREDIT"] = panel["HYG"] / panel["IEF"]
    path = DATA + r"\credit_index.parquet"
    panel.to_parquet(path)
    print(f"\nsaved {panel.shape} -> {path}\ncols: {list(panel.columns)}")
    print(panel.dropna(how="all").tail(3))
